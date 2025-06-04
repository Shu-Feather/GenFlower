from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from datetime import datetime

SEED = 0

# set random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# Basic settings
data_root = "/home/hsy/Lab-NN/flowers"
vis_root = "/home/hsy/Lab-NN/vis"
model_save_path = "/home/hsy/Lab-NN/model/Best_MLPAE.pkl"
loss_log_path = "/home/hsy/Lab-NN/logs/loss_log_MLPAE.csv"  
batch_size = 1
num_epochs = 100
early_stopping_patience = 5
IMG_WIDTH, IMG_HEIGHT = 24, 24

os.makedirs(vis_root, exist_ok=True)
os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)

class MLPAutoencoder(nn.Module):
    def __init__(self):
        super(MLPAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3 * IMG_WIDTH * IMG_HEIGHT, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Latent vector
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * IMG_WIDTH * IMG_HEIGHT),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 3 * IMG_WIDTH * IMG_HEIGHT)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 3, IMG_WIDTH, IMG_HEIGHT)

def plot_loss_curve(train_losses, valid_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    
    if valid_losses:
        valid_epochs = [epoch for epoch in epochs if epoch in valid_losses]
        valid_values = [valid_losses[epoch] for epoch in valid_epochs]
        plt.plot(valid_epochs, valid_values, 'r-s', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi = 300)
    plt.close()

def save_loss_log(train_losses, valid_losses, log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,valid_loss\n")
        for epoch in range(1, len(train_losses) + 1):
            train_loss = train_losses[epoch-1]
            valid_loss = valid_losses.get(epoch, "")
            f.write(f"{epoch},{train_loss},{valid_loss}\n")

if __name__ == "__main__":
    scheduler = exponential_decay(initial_learning_rate=1, decay_rate=0.9, decay_epochs=5)
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, IMG_WIDTH, IMG_HEIGHT
    )

    model = MLPAutoencoder()
    loss_fn_node = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0., lr=1.)

    min_valid_loss = float('inf')
    avg_train_loss = 10000.
    avg_valid_loss = 10000.
    early_stopping_counter = 0
    
    train_losses = []  # 记录每个epoch的训练损失
    valid_losses = {}  # 记录验证损失，key为epoch number
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_plot_path = os.path.join(vis_root, f"loss_curve_MLPAE_{timestamp}.png")

    for epoch in range(num_epochs):
        model.train()
        train_losses_epoch = []
        
        # Training all batches
        for images, _ in training_dataloader:
            images = images.to(torch.float32)
            
            # Forward pass
            outputs = model(images)
            
            # Compute the loss
            loss = loss_fn_node(outputs, images)
        
            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_epoch.append(loss.item())

        avg_train_loss = sum(train_losses_epoch) / len(train_losses_epoch)
        train_losses.append(avg_train_loss)
        
        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            valid_losses_epoch = []
            with torch.no_grad():
                for images, _ in validation_dataloader:
                    images = images.to(torch.float32)
                    
                    # Forward pass
                    outputs = model(images)
                
                    # Compute the validation loss
                    val_loss = loss_fn_node(outputs, images)
                    valid_losses_epoch.append(val_loss.item())

            avg_valid_loss = sum(valid_losses_epoch) / len(valid_losses_epoch)
            valid_losses[epoch] = avg_valid_loss

            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved new best model at epoch {epoch} with validation loss {avg_valid_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement")
                    break

        if epoch in valid_losses:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {valid_losses[epoch]:.4f}")
        else:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == num_epochs - 1 or early_stopping_counter >= early_stopping_patience:
            plot_loss_curve(train_losses, valid_losses, loss_plot_path)
            save_loss_log(train_losses, valid_losses, loss_log_path)
            print(f"Loss curve saved to {loss_plot_path}")
    
    plot_loss_curve(train_losses, valid_losses, loss_plot_path)
    save_loss_log(train_losses, valid_losses, loss_log_path)
    print(f"Training completed. Final loss curve saved to {loss_plot_path}")