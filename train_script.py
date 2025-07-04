import torch.optim as optim
import torch
from VarAutoencoder import VarAutoencoder, VAE_loss_function, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM
from utils.train import train
from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
from argparse import ArgumentParser
import torch.nn.functional as F
import os, sys

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="AE", choices=["VAE", "AE"])
args = parser.parse_args()

# seed
torch.manual_seed(0)
np.random.seed(0)

data_root = "./flowers"
vis_root = "./vis"
model_save_root = "./model"

IMG_WIDTH, IMG_HEIGHT = 96, 96

batch_size = 1
num_epochs = 100
early_stopping_patience = 5
model_class = VarAutoencoder if args.model == "VAE" else Autoencoder


model = model_class(
    encoding_dim=AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM,
)


# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  
scheduler = exponential_decay(initial_learning_rate=0.001, decay_rate=0.95, decay_epochs=1)  


training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, IMG_WIDTH, IMG_HEIGHT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Start training
train(
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    early_stopping_patience=early_stopping_patience,
    device=device,
    model_save_root=model_save_root,
    loss_fn=VAE_loss_function if args.model == "VAE" else F.mse_loss,
)







