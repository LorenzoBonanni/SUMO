import torch
from torch import nn


class AutoencoderNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        # Add Gaussian noise to input
        noise = torch.randn_like(x) * self.noise_std * self.training
        x_noisy = x + noise

        # Forward pass
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat