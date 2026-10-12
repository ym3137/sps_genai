"""Image classification backed by the trained CIFAR-10 CNN."""

import functools
import io
from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import CIFAR10_CLASSES, CNNClassifier, build_transform

WEIGHTS_PATH = Path(__file__).parent / "weights" / "cnn_cifar10.pt"


@functools.lru_cache(maxsize=1)
def load_model(weights_path: str = str(WEIGHTS_PATH)) -> CNNClassifier:
    """Load the trained network once and leave it in eval mode."""
    model = CNNClassifier(num_classes=len(CIFAR10_CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


class ImageClassifier:
    def __init__(self, weights_path: str = str(WEIGHTS_PATH)):
        self.model = load_model(weights_path)
        self.transform = build_transform()
        self.classes = CIFAR10_CLASSES

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """Classify a raw image and return the ranked class probabilities."""
        image = Image.open(io.BytesIO(image_bytes))
        # CIFAR-10 is an RGB dataset, so grayscale uploads and PNGs with an
        # alpha channel have to be converted before they match training.
        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]

        scores, indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        ranked = [
            {"class": self.classes[i], "probability": round(float(s), 4)}
            for s, i in zip(scores, indices)
        ]
        return {
            "predicted_class": ranked[0]["class"],
            "confidence": ranked[0]["probability"],
            "top_k": ranked,
        }
