# test/conftest.py
"""Minimal pytest configuration for SKINDX test suite."""

import pytest


@pytest.fixture
def sample_image():
    """Create a simple RGB test image."""
    from PIL import Image

    return Image.new("RGB", (224, 224), color="red")
