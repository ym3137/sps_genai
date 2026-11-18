"""Denoising diffusion model for CIFAR-10: a UNet trained to predict noise."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
NUM_CHANNELS = 3
NOISE_EMBEDDING_SIZE = 32


class SinusoidalEmbedding(nn.Module):
    """Turns a scalar noise level into a vector of sines and cosines.

    A single number carries almost no signal for a convolutional network, so the
    noise level is spread across `2 * num_frequencies` channels whose periods
    range over three orders of magnitude. Nearby noise levels then map to nearby
    embeddings while distant ones stay far apart.
    """

    def __init__(self, num_frequencies: int = NOISE_EMBEDDING_SIZE // 2):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)],
            dim=-1,
        )


class ResidualBlock(nn.Module):
    """Two convolutions plus a skip connection, with a 1x1 projection when the
    channel count changes so the skip can still be added."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.norm(x)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks at one resolution, then halve it.

    Every block output is pushed onto `skips` so the matching up block can
    recover the detail that pooling throws away.
    """

    def __init__(self, width: int, block_depth: int, in_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    """Double the resolution, then merge in the stored skips one block at a time."""

    def __init__(self, width: int, block_depth: int, in_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise present in a noisy image, conditioned on the noise level.

    The three down blocks take 32x32 to 4x4 while widening 32 -> 64 -> 96
    channels, two residual blocks run at the bottleneck, and the three up blocks
    climb back to 32x32 using the stored skips.
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        num_channels: int = NUM_CHANNELS,
        embedding_dim: int = NOISE_EMBEDDING_SIZE,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim

        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        # Starting the output layer at zero means the first predictions are zero
        # noise, which is a calm place for training to begin.
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, noisy_images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.initial(noisy_images)

        # The embedding is one vector per image; broadcasting it over every pixel
        # lets the convolutions see the noise level at each spatial position.
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)
        noise_emb = F.interpolate(
            noise_emb, size=(self.image_size, self.image_size), mode="nearest"
        )
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


def offset_cosine_schedule(
    diffusion_times: torch.Tensor,
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
) -> tuple:
    """Map a time in [0, 1] to how much signal and how much noise it carries.

    Signal and noise rates are a cosine and a sine of the same angle, so their
    squares always sum to one and the noisy image keeps unit variance. The offset
    stops the schedule at 0.95 and 0.02 rather than 1 and 0: the endpoints carry
    almost no training signal and dividing by a signal rate near zero is unstable.
    """
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)

    angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(angles), torch.cos(angles)


class DiffusionModel(nn.Module):
    """Wraps the UNet with the noise schedule, an EMA copy and the sampling loop."""

    def __init__(self, network: UNet, ema_decay: float = 0.999):
        super().__init__()
        self.network = network
        self.ema_network = copy.deepcopy(network)
        self.ema_network.eval()
        self.ema_decay = ema_decay

        self.register_buffer("normalizer_mean", torch.zeros(1, network.num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, network.num_channels, 1, 1))

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.normalizer_mean.copy_(mean.reshape_as(self.normalizer_mean))
        self.normalizer_std.copy_(std.reshape_as(self.normalizer_std))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def update_ema(self) -> None:
        """Blend the live weights into the averaged copy that generation uses.

        Sampling follows a long chain, so the jitter in the live weights compounds
        across steps; the average is both steadier and visibly cleaner.
        """
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1.0 - self.ema_decay)
            for ema_buf, buf in zip(self.ema_network.buffers(), self.network.buffers()):
                ema_buf.copy_(buf)

    def denoise(self, noisy_images, noise_rates, signal_rates, training: bool):
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)
        # Rearranged from x_t = signal * x_0 + noise * eps, which is what lets the
        # model's noise estimate be read as an estimate of the clean image.
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise: torch.Tensor, diffusion_steps: int) -> torch.Tensor:
        """Walk from pure noise back to an image in `diffusion_steps` jumps.

        Each step predicts the clean image and the noise, then re-mixes the two at
        the next, lower noise level. Because the re-mix reuses the predicted noise
        instead of drawing fresh noise, the trajectory is deterministic and a
        handful of steps already gives a usable image.
        """
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device)
            t = t * (1.0 - step * step_size)

            noise_rates, signal_rates = offset_cosine_schedule(t)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )

            next_noise_rates, next_signal_rates = offset_cosine_schedule(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    @torch.no_grad()
    def generate(self, num_images: int, diffusion_steps: int = 20, generator=None) -> torch.Tensor:
        device = self.normalizer_mean.device
        shape = (num_images, self.network.num_channels, self.network.image_size, self.network.image_size)
        initial_noise = torch.randn(shape, device=device, generator=generator)
        return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))
