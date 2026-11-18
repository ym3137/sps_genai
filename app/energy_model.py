"""Energy-based model for CIFAR-10, trained with contrastive divergence."""

import torch
import torch.nn as nn

IMAGE_SIZE = 32
NUM_CHANNELS = 3


def swish(x: torch.Tensor) -> torch.Tensor:
    """Smooth stand-in for ReLU used across the energy network.

    Sampling differentiates the energy with respect to the pixels, so a ReLU's
    flat half would hand back a zero gradient over much of the input and the
    sampler would stall there.
    """
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Scores an image with a single scalar: low energy for real, high for fake.

    Four stride-2 convolutions take the 32x32 input down to 2x2 while widening
    the channels, and two linear layers collapse that to one number.
    """

    def __init__(self, num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


@torch.enable_grad()
def langevin_sample(
    model: EnergyModel,
    images: torch.Tensor,
    steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
) -> torch.Tensor:
    """Walk `images` downhill on the energy surface with Langevin dynamics.

    The gradient is taken with respect to the pixels rather than the weights, so
    the network stays fixed while the sample moves. Noise is injected at every
    step to keep the chain exploring instead of collapsing onto one minimum, and
    both the gradient and the pixels are clamped so the chain cannot diverge.

    The decorator matters: callers run inference under `torch.no_grad()`, and
    without it the gradient the sampler depends on would never be built.
    """
    was_training = model.training
    model.eval()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(images) * noise_std
            images = (images + noise).clamp(-1.0, 1.0)

        images.requires_grad_(True)
        energy = model(images)
        grads, = torch.autograd.grad(
            energy, images, grad_outputs=torch.ones_like(energy)
        )

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            images = (images - step_size * grads).clamp(-1.0, 1.0)

    model.train(was_training)
    return images.detach()
