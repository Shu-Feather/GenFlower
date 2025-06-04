import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 24, 24

class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        # 3 * H * W -> 32 * H/2 * W/2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 32 * H/2 * W/2 -> 64 * H/4 * W/4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 64 * H/4 * W/4 -> 128 * H/4 * W/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc = nn.Linear(128 * (IMG_HEIGHT//4) * (IMG_WIDTH//4), encoding_dim)

    def forward(self, x):
        # 3 * H * W -> 32 * H/2 * W/2
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 32 * H/2 * W/2 -> 64 * H/4 * W/4
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 64 * H/4 * W/4 -> 128 * H/4 * W/4
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        v = self.fc(x)
        return v


class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(encoding_dim, 128 * (IMG_HEIGHT//4) * (IMG_WIDTH//4))
        
        # 128 * H/4 * W/4 -> 64 * H/4 * W/4 
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        # 64 * H/4 * W/4 -> 32 * H/2 * W/2
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 32 * H/2 * W/2 -> 3 * H * W 
        self.upconv2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, v):
        # latent vector -> 128 * H/4 * W/4
        x = self.fc(v)
        x = x.view(x.size(0), 128, IMG_HEIGHT//4, IMG_WIDTH//4)
        
        # 128 * H/4 * W/4 -> 64 * H/4 * W/4
        x = F.relu(self.conv1(x))
        
        # 64 * H/4 * W/4 -> 32 * H/2 * W/2
        x = F.relu(self.upconv1(x))
        
        # 32 * H/2 * W/2 -> 3 * H * W 
        x = torch.sigmoid(self.upconv2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        v = self.encoder(x)
        x_recon = self.decoder(v)
        return x_recon
    
    @property
    def name(self):
        return "AE"