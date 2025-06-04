import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math
import os

torch.manual_seed(0)
np.random.seed(0)

model_save_path = "/home/hsy/Lab-NN/model/Best_Diffusion_Model.pkl"
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 96, 96

num_steps = 1000
beta_min = 1e-4
beta_max = 0.02

def create_diffusion_params(device):
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

class TimeEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        pos_enc = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        groups1 = 1 if in_channels < 8 or in_channels % 8 != 0 else 8
        groups2 = 1 if out_channels < 8 or out_channels % 8 != 0 else 8
        
        self.block1 = torch.nn.Sequential(
            torch.nn.GroupNorm(groups1, in_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = torch.nn.Linear(time_dim, out_channels)
        
        self.block2 = torch.nn.Sequential(
            torch.nn.GroupNorm(groups2, out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_steps, img_channels=3, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4
        
        self.time_embed = torch.nn.Sequential(
            TimeEmbedding(base_dim),
            torch.nn.Linear(base_dim, time_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, time_dim)
        )
        
        self.down1 = ResidualBlock(img_channels, base_dim, time_dim)
        self.down2 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.down3 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        
        self.mid = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)
        
        self.up3 = ResidualBlock(base_dim * 8, base_dim * 2, time_dim)
        self.up2 = ResidualBlock(base_dim * 4, base_dim, time_dim)
        self.up1 = ResidualBlock(base_dim * 2, base_dim, time_dim)
        
        self.out_conv = torch.nn.Conv2d(base_dim, img_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        
        m = self.mid(d3, t_emb)
        
        u3 = self.up3(torch.cat([d3, m], dim=1), t_emb)
        u2 = self.up2(torch.cat([d2, u3], dim=1), t_emb)
        u1 = self.up1(torch.cat([d1, u2], dim=1), t_emb)
        
        return self.out_conv(u1)

def p_theta_sampling(model, x, t, diffusion_params):
    with torch.no_grad():
        pred_noise = model(x, t)
        
        t_idx = t[0] if t.dim() > 0 else t
        alpha_t = diffusion_params['alphas'][t_idx].view(-1, 1, 1, 1)
        alpha_bar_t = diffusion_params['alphas_bar'][t_idx].view(-1, 1, 1, 1)
        beta_t = diffusion_params['betas'][t_idx].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = diffusion_params['sqrt_one_minus_alphas_bar'][t_idx].view(-1, 1, 1, 1)
        
        mean = (x - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / torch.sqrt(alpha_t)
        
        if t_idx > 0:
            var = diffusion_params['posterior_variance'][t_idx].view(-1, 1, 1, 1)
            z = torch.randn_like(x)
            return mean + torch.sqrt(var) * z
        else:
            return mean

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

def generate_and_save_images(model, num_images, device, diffusion_params):
    sample_shape = (num_images, 3, IMG_WIDTH, IMG_HEIGHT)
    
    x_seq = p_theta_sampling_loop(model, sample_shape, num_steps, device, diffusion_params)
    generated_images = x_seq[-1] 
    
    generated_images = (generated_images + 1) / 2
    
    for i in range(num_images):
        save_image(generated_images[i], f"{output_dir}/generated_image_{i+1}.png")
    
    save_image(generated_images, f"{output_dir}/image_grid.png", nrow=4)
    


    plt.figure(figsize=(10, 10))
    grid = torch.cat([generated_images[i] for i in range(num_images)], dim=1)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f'Generated Images (n={num_images})')
    plt.savefig(f"{output_dir}/image_grid_plot.png", bbox_inches='tight')
    plt.show()
    
    return generated_images

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    diffusion_params = create_diffusion_params(device)
    
    model = UNet(num_steps).to(device)
    
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print(f"Loaded trained model from {model_save_path}")
    
    num_images = 36
    
    generated_images = generate_and_save_images(model, num_images, device, diffusion_params)
    
    print(f"Successfully generated {num_images} images in {output_dir}")