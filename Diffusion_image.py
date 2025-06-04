from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "/home/hsy/Lab-NN/flowers_full"
model_save_path = "/home/hsy/Lab-NN/model/Best_Diffusion_Model.pkl"
vis_dir = "./vis/diffusion"
os.makedirs(vis_dir, exist_ok=True)

# Hyperparameters
batch_size = 8
num_epochs = 200
learning_rate = 1e-4
early_stopping_patience = 30
IMG_WIDTH, IMG_HEIGHT = 32, 32

# Diffusion process settings
num_steps = 1000
beta_min = 1e-4
beta_max = 0.02

def create_diffusion_params(device):
    # Cosine noise schedule (Improved DDPM)
    def cosine_beta_schedule(n_steps, s=0.008):
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps)
        alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    betas = cosine_beta_schedule(num_steps).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_bar': sqrt_alphas_bar,
        'sqrt_one_minus_alphas_bar': sqrt_one_minus_alphas_bar,
        'posterior_variance': posterior_variance
    }

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        pos_enc = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)

# Residual Block with safe GroupNorm
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        # Calculate safe group numbers
        groups1 = 1 if in_channels < 8 or in_channels % 8 != 0 else 8
        groups2 = 1 if out_channels < 8 or out_channels % 8 != 0 else 8
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups2, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

# U-Net Model
class UNet(nn.Module):
    def __init__(self, n_steps, img_channels=3, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Downsample blocks
        self.down1 = ResidualBlock(img_channels, base_dim, time_dim)
        self.down2 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.down3 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        
        # Middle block
        self.mid = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)
        
        # Upsample blocks
        self.up3 = ResidualBlock(base_dim * 8, base_dim * 2, time_dim)
        self.up2 = ResidualBlock(base_dim * 4, base_dim, time_dim)
        self.up1 = ResidualBlock(base_dim * 2, base_dim, time_dim)
        
        # Output convolution
        self.out_conv = nn.Conv2d(base_dim, img_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Downsample path
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        
        # Middle block
        m = self.mid(d3, t_emb)
        
        # Upsample path with skip connections
        u3 = self.up3(torch.cat([d3, m], dim=1), t_emb)
        u2 = self.up2(torch.cat([d2, u3], dim=1), t_emb)
        u1 = self.up1(torch.cat([d1, u2], dim=1), t_emb)
        
        return self.out_conv(u1)

# Loss function with device-aware diffusion parameters
def diffusion_loss_fn(model, x_0, n_steps, diffusion_params):
    batch_size = x_0.size(0)
    t = torch.randint(0, n_steps, (batch_size,), device=x_0.device).long()
    
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = diffusion_params['sqrt_alphas_bar'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = diffusion_params['sqrt_one_minus_alphas_bar'][t].view(-1, 1, 1, 1)
    
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    predicted_noise = model(x_t, t)
    
    return nn.MSELoss()(predicted_noise, noise)

# Sampling function with device-aware diffusion parameters
def p_theta_sampling(model, x, t, diffusion_params):
    with torch.no_grad():
        # Predict noise
        pred_noise = model(x, t)
        
        # Calculate coefficients
        t_idx = t[0] if t.dim() > 0 else t
        alpha_t = diffusion_params['alphas'][t_idx].view(-1, 1, 1, 1)
        alpha_bar_t = diffusion_params['alphas_bar'][t_idx].view(-1, 1, 1, 1)
        beta_t = diffusion_params['betas'][t_idx].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = diffusion_params['sqrt_one_minus_alphas_bar'][t_idx].view(-1, 1, 1, 1)
        
        # Calculate mean
        mean = (x - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / torch.sqrt(alpha_t)
        
        # Calculate variance
        if t_idx > 0:
            var = diffusion_params['posterior_variance'][t_idx].view(-1, 1, 1, 1)
            z = torch.randn_like(x)
            return mean + torch.sqrt(var) * z
        else:
            return mean

# Sampling loop with device-aware diffusion parameters
@torch.no_grad()
def p_theta_sampling_loop(model, shape, n_steps, device, diffusion_params):
    model.eval()
    x_t = torch.randn(shape, device=device)
    x_seq = [x_t]
    
    for t in reversed(range(n_steps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_t = p_theta_sampling(model, x_t, t_tensor, diffusion_params)
        x_seq.append(x_t)
    
    return x_seq

# Visualization function with device-aware diffusion parameters
def visualize_sampling(model, sample_shape, n_steps, epoch, device, diffusion_params):
    model.eval()
    x_seq = p_theta_sampling_loop(model, sample_shape, n_steps, device, diffusion_params)
    
    # Save final image
    final_img = x_seq[-1][0].detach().cpu()
    final_img = (final_img + 1) / 2  # [-1,1] to [0,1]
    save_image(final_img, f"{vis_dir}/sample_epoch_{epoch}.png")
    
    # Create and save progress grid
    num_shows = 10
    step_interval = n_steps // num_shows
    fig, axes = plt.subplots(1, num_shows, figsize=(20, 2))
    for i, ax in enumerate(axes):
        step = i * step_interval
        image = x_seq[step][0].detach().cpu().permute(1, 2, 0)
        image = (image + 1) / 2  # Normalize to [0,1]
        ax.imshow(image.clip(0, 1))
        ax.axis('off')
        ax.set_title(f"Step {step}")
    plt.suptitle(f"Sampling Visualization - Epoch {epoch}")
    plt.savefig(f"{vis_dir}/progress_epoch_{epoch}.png", bbox_inches='tight')
    plt.close()

# Loss tracking
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{vis_dir}/loss_curve.png")
    plt.close()

if __name__ == "__main__":
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create diffusion parameters on the correct device
    diffusion_params = create_diffusion_params(device)
    
    # Create dataloaders
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, IMG_WIDTH, IMG_HEIGHT
    )

    # Initialize model and optimizer
    model = UNet(num_steps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # EMA for better sampling
    ema_model = UNet(num_steps).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.995

    min_valid_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    sample_shape = (1, 3, IMG_WIDTH, IMG_HEIGHT)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for images, _ in training_dataloader:
            images = images.to(device).float() * 2 - 1  # Normalize to [-1, 1]
            
            loss = diffusion_loss_fn(model, images, num_steps, diffusion_params)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            epoch_train_loss += loss.item()
            
            # Update EMA model
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        avg_train_loss = epoch_train_loss / len(training_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, _ in validation_dataloader:
                images = images.to(device).float() * 2 - 1
                loss = diffusion_loss_fn(model, images, num_steps, diffusion_params)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(validation_dataloader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        # Print stats
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Visualize sampling every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_sampling(ema_model, sample_shape, num_steps, epoch+1, device, diffusion_params)
        
        # Early stopping and model saving
        if avg_val_loss < min_valid_loss:
            min_valid_loss = avg_val_loss
            torch.save(ema_model.state_dict(), model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot loss curves
    plot_losses(train_losses, val_losses)
    
    # Final visualization
    model.load_state_dict(torch.load(model_save_path))
    visualize_sampling(model, (16, 3, IMG_WIDTH, IMG_HEIGHT), num_steps, "final", device, diffusion_params)