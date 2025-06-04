import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 96, 96

# Define the Variational Encoder 
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers to generate mu and log_var
        self.flat_features = 128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4) 
        self.fc_mu = nn.Linear(self.flat_features, encoding_dim)
        self.fc_log_var = nn.Linear(self.flat_features, encoding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.flat_features) 
        
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarDecoder, self).__init__()
        self.flat_features = 128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4)
        self.fc = nn.Linear(encoding_dim, self.flat_features)  
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, v):
        x = F.relu(self.fc(v))
        x = x.view(-1, 128, IMG_HEIGHT//4, IMG_WIDTH//4)  
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  
        
        return x

# Define the Variational Autoencoder (VAE)
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise from standard normal distribution
        z = mu + eps * std             # Sampled latent vector
        return z
        
    def forward(self, x):
        # Encode input to get mu and log_var
        mu, log_var = self.encoder(x)
        # Sample latent vector using reparameterization
        z = self.reparameterize(mu, log_var)
        # Decode to reconstruct image
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    # Unpack the outputs
    x_reconstructed, mu, log_var = outputs
    
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(x_reconstructed, images, reduction='sum')
    
    # KL divergence loss (closed-form solution)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss = reconstruction loss + KL divergence
    total_loss = reconstruction_loss + kl_divergence
    
    return total_loss