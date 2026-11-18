"""CIFAR-10 image generation from the trained diffusion and energy models."""

import functools
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.diffusion_model import DiffusionModel, UNet
from app.energy_model import EnergyModel, langevin_sample

WEIGHTS_DIR = Path(__file__).parent / "weights"
DIFFUSION_WEIGHTS = WEIGHTS_DIR / "diffusion_cifar10.pt"
ENERGY_WEIGHTS = WEIGHTS_DIR / "energy_cifar10.pt"


@functools.lru_cache(maxsize=1)
def load_diffusion_model(weights_path: str = str(DIFFUSION_WEIGHTS)) -> DiffusionModel:
    """Load the trained diffusion model once and leave it in eval mode."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    model = DiffusionModel(UNet())
    # Only the averaged weights are shipped, so both copies start from them and
    # `denoise(training=False)` reads the same network either way.
    model.network.load_state_dict(checkpoint["ema_network"])
    model.ema_network.load_state_dict(checkpoint["ema_network"])
    model.set_normalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def load_energy_model(weights_path: str = str(ENERGY_WEIGHTS)) -> EnergyModel:
    """Load the trained energy network once and leave it in eval mode."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    model = EnergyModel()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _seeded_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert a batch of [0, 1] float images to (N, 32, 32, 3) uint8."""
    scaled = (images * 255.0).clamp(0, 255).to(torch.uint8)
    return scaled.permute(0, 2, 3, 1).numpy()


class DiffusionImageGenerator:
    """Samples CIFAR-10 images by reversing the diffusion process from noise."""

    def __init__(self, weights_path: str = str(DIFFUSION_WEIGHTS)):
        self.model = load_diffusion_model(weights_path)

    def sample(self, count: int = 8, steps: int = 20, seed: int | None = None) -> np.ndarray:
        """Generate `count` images using `steps` reverse-diffusion jumps.

        More steps follow the schedule more finely at a proportional cost in
        time; 20 is already enough for the deterministic sampler used here.
        """
        images = self.model.generate(
            count, diffusion_steps=steps, generator=_seeded_generator(seed)
        )
        return _to_uint8(images)


class EnergyImageGenerator:
    """Samples CIFAR-10 images by running Langevin dynamics on the energy surface."""

    def __init__(self, weights_path: str = str(ENERGY_WEIGHTS)):
        self.model = load_energy_model(weights_path)

    def sample(
        self,
        count: int = 8,
        steps: int = 128,
        step_size: float = 10.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Start from uniform noise and walk downhill on the learned energy."""
        generator = _seeded_generator(seed)
        noise = torch.rand(count, 3, 32, 32, generator=generator) * 2 - 1
        samples = langevin_sample(self.model, noise, steps=steps, step_size=step_size)
        # The energy model works in [-1, 1]; shift back to [0, 1] for display.
        return _to_uint8((samples + 1.0) / 2.0)

    def energy(self, images: torch.Tensor) -> torch.Tensor:
        """Score images with the learned energy, lower meaning more realistic."""
        with torch.no_grad():
            return self.model(images)


def to_png(image: np.ndarray, scale: int = 4) -> bytes:
    img = Image.fromarray(image, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_grid(images: np.ndarray, columns: int = 8, scale: int = 3, pad: int = 2) -> bytes:
    """Tile the samples into a single PNG for quick visual inspection."""
    n, h, w, c = images.shape
    columns = max(1, min(columns, n))
    rows = (n + columns - 1) // columns

    canvas = np.zeros((rows * (h + pad) + pad, columns * (w + pad) + pad, c), dtype=np.uint8)
    for i, image in enumerate(images):
        r, col = divmod(i, columns)
        y, x = pad + r * (h + pad), pad + col * (w + pad)
        canvas[y:y + h, x:x + w] = image

    img = Image.fromarray(canvas, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
