"""Hand-written digit generation using the trained GAN generator."""

import functools
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.gan_model import LATENT_DIM, Generator

WEIGHTS_PATH = Path(__file__).parent / "weights" / "gan_generator.pt"


@functools.lru_cache(maxsize=1)
def load_generator(weights_path: str = str(WEIGHTS_PATH)) -> Generator:
    """Load the trained generator once and leave it in eval mode."""
    model = Generator()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


class DigitGenerator:
    def __init__(self, weights_path: str = str(WEIGHTS_PATH)):
        self.model = load_generator(weights_path)
        self.latent_dim = LATENT_DIM

    def sample(self, count: int = 16, seed: int | None = None) -> np.ndarray:
        """Generate `count` digits as a (count, 28, 28) uint8 array.

        Passing a seed makes the output reproducible, which is what lets the
        same request be replayed and compared.
        """
        if seed is None:
            z = torch.randn(count, self.latent_dim)
        else:
            gen = torch.Generator().manual_seed(seed)
            z = torch.randn(count, self.latent_dim, generator=gen)

        with torch.no_grad():
            images = self.model(z)

        # The generator ends in Tanh, so its output is in [-1, 1].
        scaled = (images + 1.0) * 127.5
        return scaled.clamp(0, 255).to(torch.uint8).numpy()[:, 0]

    @staticmethod
    def to_png(image: np.ndarray, scale: int = 4) -> bytes:
        img = Image.fromarray(image, mode="L")
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def to_grid(images: np.ndarray, columns: int = 8, scale: int = 3,
                pad: int = 2) -> bytes:
        """Tile the digits into a single PNG for quick visual inspection."""
        n, h, w = images.shape
        columns = max(1, min(columns, n))
        rows = (n + columns - 1) // columns

        canvas = np.zeros((rows * (h + pad) + pad, columns * (w + pad) + pad),
                          dtype=np.uint8)
        for i, im in enumerate(images):
            r, c = divmod(i, columns)
            y, x = pad + r * (h + pad), pad + c * (w + pad)
            canvas[y:y + h, x:x + w] = im

        img = Image.fromarray(canvas, mode="L")
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
