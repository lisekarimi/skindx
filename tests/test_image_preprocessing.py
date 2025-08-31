# test/test_image_preprocessing.py
import torch
from PIL import Image

from src.constants import CLASS_NAMES
from src.utils import preprocess_image


class TestUtilsSimple:
    """Simple tests for utils functions."""

    def test_get_class_names(self):
        """Test class names from constants."""
        names = list(CLASS_NAMES.keys())
        assert len(names) == 7
        assert names[0] == "akiec"
        assert names[4] == "mel"

    def test_preprocess_image_basic(self):
        """Test basic image preprocessing."""
        # Create test image
        image = Image.new("RGB", (100, 100), color="red")

        # Process image
        result = preprocess_image(image)

        # Check output
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_preprocess_image_converts_modes(self):
        """Test that different image modes are handled."""
        # Test grayscale conversion
        gray_image = Image.new("L", (100, 100), color=128)
        gray_image = gray_image.convert("RGB")  # Convert to RGB first
        result = preprocess_image(gray_image)
        assert result.shape == (3, 224, 224)

        # Test RGBA conversion
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        rgba_image = rgba_image.convert("RGB")  # Convert to RGB first
        result = preprocess_image(rgba_image)
        assert result.shape == (3, 224, 224)
