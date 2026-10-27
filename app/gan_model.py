"""DCGAN for MNIST digit generation.

The architecture follows the Assignment 3 specification exactly.

    Generator                                   Discriminator
    noise (B, 100)                              image (B, 1, 28, 28)
    Linear -> 7 x 7 x 128, reshape              Conv2d  1 -> 64,  k4 s2 p1  -> 14 x 14
    ConvTranspose2d 128 -> 64, k4 s2 p1         LeakyReLU(0.2)
      -> 14 x 14, BatchNorm2d, ReLU             Conv2d 64 -> 128, k4 s2 p1  ->  7 x 7
    ConvTranspose2d  64 ->  1, k4 s2 p1           BatchNorm2d, LeakyReLU(0.2)
      -> 28 x 28, Tanh                          Flatten, Linear -> 1
"""

import torch.nn as nn

LATENT_DIM = 100
IMAGE_SIZE = 28


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.project = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.project(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        """Return the raw score. Sigmoid is applied inside BCEWithLogitsLoss."""
        return self.classifier(self.features(x))
