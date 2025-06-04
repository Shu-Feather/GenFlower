import torch
from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--model", type=str, default="VAE", choices=["VAE", "AE"])
args = parser.parse_args()

# Set paths and model configuration based on the selected model
model_save_path = f"./model/Best_{args.model}.pth"
vis_root = "./vis"
model_class = VarAutoencoder if args.model == "VAE" else Autoencoder
ENCODING_DIM = AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM

# Initialize the selected model and load its parameters
model = model_class(encoding_dim=ENCODING_DIM)
model.load_state_dict(torch.load(model_save_path))

# Ensure the model is in evaluation mode
model.eval()

# TODO: Generate random images
'''
Steps:
1. Sample 10 latent vectors from a standard normal distribution (mean=0, std=1).
2. Pass the sampled latent vectors through the decoder to generate images.
3. Ensure the generated images are stored in a tensor `random_images` with shape (10, 3, 24, 24).
'''

# Step 1: Sample latent vectors from a standard normal distribution
latent_vectors = torch.randn(10, ENCODING_DIM)  # Shape: (10, encoding_dim)

# Step 2: Use the decoder to generate images
with torch.no_grad():  # No gradients needed for generation
    random_images = model.decoder(latent_vectors)  # Shape: (10, 3, 24, 24)

# Step 3: Ensure the output images are in the valid range [0, 1]
random_images = torch.clamp(random_images, 0, 1)

# Save the 10 random images in one figure
fig = plt.figure(figsize=(10, 1))

for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    # Convert tensor to numpy array and transpose dimensions for visualization
    ax.imshow(np.transpose(random_images[i].detach().numpy(), (1, 2, 0)))

# Save the generated images
plt.savefig(f"{vis_root}/random_images_{args.model}.png")