"""CNN image classifier for CIFAR-10.

The architecture follows the Assignment 2 specification:

    Input            64 x 64 x 3
    Conv2D           16 filters, 3x3, stride 1, padding 1
    ReLU
    MaxPool2D        2x2, stride 2
    Conv2D           32 filters, 3x3, stride 1, padding 1
    ReLU
    MaxPool2D        2x2, stride 2
    Flatten
    Linear           100 units
    ReLU
    Linear           10 units
"""

import torch.nn as nn
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

INPUT_SIZE = 64

# Channel means and standard deviations of the CIFAR-10 training set.
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)


def build_transform(input_size: int = INPUT_SIZE):
    """Preprocessing used at both training and inference time.

    Keeping this in one place is what guarantees an uploaded image is
    normalised exactly the way the training images were.
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, input_size: int = INPUT_SIZE):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Each pooling layer halves both spatial dimensions, so a 64x64 input
        # reaches the first linear layer as 32 channels of 16x16.
        spatial = input_size // 4
        self.fc1 = nn.Linear(32 * spatial * spatial, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
