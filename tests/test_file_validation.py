# test/test_file_validation.py
from unittest.mock import Mock

from src.ui.ui_utils import validate_uploaded_file


class TestFileValidation:
    """Test file validation with mocks only."""

    def test_no_file(self):
        """Test with no file."""
        is_valid, message = validate_uploaded_file(None)
        assert is_valid is False
        assert message == "No file uploaded"

    def test_valid_file(self):
        """Test with valid mocked file."""
        mock_file = Mock()
        mock_file.name = "test.jpg"
        mock_file.type = "image/jpeg"
        mock_file.size = 1024 * 1024  # 1MB
        mock_file.seek = Mock()

        is_valid, message = validate_uploaded_file(mock_file)
        assert is_valid is True
        assert message == "File is valid"

    def test_file_too_large(self):
        """Test oversized file."""
        mock_file = Mock()
        mock_file.name = "large.jpg"
        mock_file.type = "image/jpeg"
        mock_file.size = 15 * 1024 * 1024  # 15MB

        is_valid, message = validate_uploaded_file(mock_file)
        assert is_valid is False
        assert message == "File size exceeds 10MB limit"
