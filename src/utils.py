# src/utils.py

import torch
from PIL import Image
from torchvision import transforms


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference.

    Same preprocessing as used during training.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transform(image)
