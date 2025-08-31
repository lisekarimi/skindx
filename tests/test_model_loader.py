# tests/test_model_loader.py

import logging
from unittest.mock import Mock, patch

import pytest
import requests
import torch
from PIL import Image

from src.model_loader import ModelLoader

NUM_CLASSES = 7  # avoid constructing ModelLoader() in tests


# -------------------- Basic tests (happy path + simple errors) --------------------


class TestModelLoaderSimple:
    """Minimal test suite for ModelLoader."""

    @pytest.fixture
    def sample_image(self):
        """Return a new PIL Image object with dimensions (224, 224) and color 'red'.

        Returns
        -------
            PIL.Image.Image: A red image of size 224x224 pixels.

        """
        return Image.new("RGB", (224, 224), color="red")

    @patch("src.model_loader.ModelLoader._warmup_model")
    @patch("src.model_loader.ModelLoader._validate_model")
    @patch("src.model_loader.hf_hub_download")
    @patch("src.model_loader.torch.load")
    @patch("torch.cuda.is_available", return_value=False)
    def test_model_loader_init_basic(
        self, _mock_cuda, mock_torch_load, mock_hf_download, mock_validate, mock_warmup
    ):
        """Test basic ModelLoader initialization with mocked dependencies.

        Verifies that:
        - Model is loaded successfully
        - Model path is set correctly
        - Validation and warmup methods are called
        """
        mock_hf_download.return_value = "/cache/model.pth"
        mock_torch_load.return_value = Mock()
        loader = ModelLoader()
        assert loader.model is not None
        assert loader.model_path == "/cache/model.pth"
        mock_validate.assert_called_once()
        mock_warmup.assert_called_once()

    @patch("src.model_loader.hf_hub_download")
    def test_download_retry_repository_not_found(self, mock_hf_download):
        """Test that repository not found errors are properly handled."""
        mock_hf_download.side_effect = Exception("Repository not found")
        with pytest.raises(ValueError, match="not found or access denied"):
            ModelLoader()

    def test_predict_no_model_loaded(self, sample_image):
        """Test that predict raises ValueError when no model is loaded."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.model = None
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.predict(sample_image)


# -------------- Extended coverage (normal path + warmup/validate) -----------------


class TestModelLoaderExtended:
    """Extended tests for ModelLoader to improve coverage."""

    @patch("src.model_loader.hf_hub_download")
    def test_download_retry_rate_limited(self, mock_hf_download):
        """Test that rate limited requests are retried and eventually succeed."""

        class R:
            status_code = 429

        mock_hf_download.side_effect = [
            requests.HTTPError(response=R()),
            "/tmp/model.pth",
        ]
        loader = ModelLoader.__new__(ModelLoader)
        with (
            patch("src.model_loader.torch.load", return_value=Mock()),
            patch.object(loader, "_load_model", return_value=None),
        ):
            path = loader._download_with_retry("repo", "file", delay=0)
            assert path == "/tmp/model.pth"

    @patch("src.model_loader.torch.load")
    def test_load_model_and_validate(self, mock_torch_load):
        """Test model load and validation with mocked deps."""
        fake_model = Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_model.return_value = torch.randn(1, NUM_CLASSES)
        mock_torch_load.return_value = fake_model

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_path = "/tmp/model.pth"
        loader.device = torch.device("cpu")
        loader.class_names = list(range(NUM_CLASSES))

        with patch.object(loader, "_warmup_model", return_value=None):
            loader._load_model()
        assert loader.model is not None

    def test_validate_model_shape_mismatch(self):
        """Test validation fails when output shape mismatches classes."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.model = lambda x: torch.randn(1, 99)  # mismatch
        loader.class_names = list(range(7))
        with pytest.raises(ValueError, match="doesn't match expected classes"):
            loader._validate_model()

    def test_warmup_model_runs(self):
        """Test that model warmup runs successfully without raising exceptions."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.model = lambda x: torch.randn(x.shape[0], 7)
        loader.class_names = list(range(7))
        loader._warmup_model()  # no exception

    def test_predict_success(self):
        """Test predict returns result with all required fields."""
        img = Image.new("RGB", (224, 224), color="blue")
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.class_names = ["a", "b", "c", "d", "e", "f", "g"]
        loader.model = lambda x: torch.randn(x.shape[0], len(loader.class_names))
        result = loader.predict(img)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result


# -------------------- Edge/error branches --------------------


class TestModelLoaderEdgeCases:
    """Error branches for retries, warnings, and failure paths."""

    # _download_with_retry branches

    @patch("src.model_loader.hf_hub_download")
    def test_retry_401_auth_error(self, mock_hf):
        """Test that 401 authentication errors are properly converted to ValueError."""

        class R:
            status_code = 401

        mock_hf.side_effect = requests.HTTPError(response=R())
        loader = ModelLoader.__new__(ModelLoader)
        with pytest.raises(ValueError, match="Authentication failed"):
            loader._download_with_retry("repo", "file", max_retries=1, delay=0)

    @patch("src.model_loader.hf_hub_download")
    def test_retry_500_then_success(self, mock_hf):
        """Test that 500 errors are retried and eventually succeed after retry."""

        class R:
            status_code = 500

        mock_hf.side_effect = [requests.HTTPError(response=R()), "/tmp/model.pth"]
        loader = ModelLoader.__new__(ModelLoader)
        with (
            patch("src.model_loader.torch.load", return_value=Mock()),
            patch.object(loader, "_load_model", return_value=None),
        ):
            assert (
                loader._download_with_retry("repo", "file", delay=0) == "/tmp/model.pth"
            )

    @patch("src.model_loader.hf_hub_download")
    def test_retry_connection_then_success(self, mock_hf):
        """Test connection errors retry and eventually succeed."""
        mock_hf.side_effect = [ConnectionError("boom"), "/tmp/model.pth"]
        loader = ModelLoader.__new__(ModelLoader)
        with (
            patch("src.model_loader.torch.load", return_value=Mock()),
            patch.object(loader, "_load_model", return_value=None),
        ):
            assert (
                loader._download_with_retry("repo", "file", delay=0) == "/tmp/model.pth"
            )

    @patch("src.model_loader.hf_hub_download")
    def test_retry_exhausted_network(self, mock_hf):
        """Test connection errors raise after max retries."""
        mock_hf.side_effect = [
            ConnectionError("a"),
            ConnectionError("b"),
            ConnectionError("c"),
        ]
        loader = ModelLoader.__new__(ModelLoader)
        with pytest.raises(ConnectionError):
            loader._download_with_retry("repo", "file", max_retries=3, delay=0)

    @patch("src.model_loader.hf_hub_download")
    def test_repo_access_denied_maps_valueerror(self, mock_hf):
        """Test repo access denied errors convert to ValueError."""
        mock_hf.side_effect = Exception("Access denied to repo")
        loader = ModelLoader.__new__(ModelLoader)
        with pytest.raises(ValueError, match="access denied"):
            loader._download_with_retry("repo", "file", max_retries=1, delay=0)

    # _validate_model branches

    def test_validate_logs_probabilities_warning(self, caplog):
        """Test validation warns when outputs are probabilities, not logits."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.class_names = list(range(7))
        loader.model = lambda x: torch.ones(1, len(loader.class_names))  # non-negative
        caplog.set_level(logging.WARNING)
        loader._validate_model()  # no raise
        assert any("probabilities" in r.message for r in caplog.records)

    def test_validate_model_negative_logits_logs_info(self, caplog):
        """Test validation logs info when logits include negatives."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.class_names = list(range(7))
        loader.model = lambda x: torch.tensor([[-1.0] + [0.0] * 6])  # has negative
        caplog.set_level(logging.INFO)
        loader._validate_model()
        assert any("raw logits (expected)" in r.message for r in caplog.records)

    # _warmup_model failure path

    def test_warmup_handles_exception(self, caplog):
        """Test that warmup handles exceptions gracefully and logs warning."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.class_names = list(range(7))

        def bad_model(x):
            raise RuntimeError("fail")

        loader.model = bad_model
        caplog.set_level(logging.WARNING)
        loader._warmup_model()  # should not raise
        assert any("Model warmup failed" in r.message for r in caplog.records)

    # predict() error path

    def test_predict_model_raises(self):
        """Test that predict properly propagates exceptions raised by the model."""
        loader = ModelLoader.__new__(ModelLoader)
        loader.device = torch.device("cpu")
        loader.class_names = list(range(7))

        def bad_model(x):
            raise RuntimeError("predict boom")

        loader.model = bad_model
        with pytest.raises(RuntimeError, match="predict boom"):
            loader.predict(Image.new("RGB", (10, 10), "red"))


# -------------------- More edge cases to push coverage --------------------


class TestModelLoaderMoreEdgeCases:
    """Remaining retry branches and load hooks."""

    @patch("src.model_loader.hf_hub_download")
    def test_retry_429_exhausted(self, mock_hf):
        """Test 429 errors raise ValueError after max retries."""

        class R:
            status_code = 429

        mock_hf.side_effect = [requests.HTTPError(response=R())] * 3
        loader = ModelLoader.__new__(ModelLoader)
        with pytest.raises(ValueError, match="Rate limited by HuggingFace"):
            loader._download_with_retry("repo", "file", max_retries=3, delay=0)

    @patch("src.model_loader.hf_hub_download")
    def test_retry_500_exhausted(self, mock_hf):
        """Test 502 errors raise HTTPError after max retries."""

        class R:
            status_code = 502

        mock_hf.side_effect = [requests.HTTPError(response=R())] * 2
        loader = ModelLoader.__new__(ModelLoader)
        with pytest.raises(requests.HTTPError):
            loader._download_with_retry("repo", "file", max_retries=2, delay=0)

    @patch("src.model_loader.hf_hub_download")
    def test_retry_timeout_then_success(self, mock_hf):
        """Test that timeout errors are retried and eventually succeed after retry."""
        mock_hf.side_effect = [TimeoutError("t"), "/tmp/model.pth"]
        loader = ModelLoader.__new__(ModelLoader)
        with (
            patch("src.model_loader.torch.load", return_value=Mock()),
            patch.object(loader, "_load_model", return_value=None),
        ):
            assert (
                loader._download_with_retry("repo", "file", delay=0) == "/tmp/model.pth"
            )

    @patch("src.model_loader.torch.load")
    def test_load_model_calls_to_and_eval(self, mock_torch_load):
        """Test that model loading properly calls to() and eval() methods."""
        fake = Mock()
        fake.to.return_value = fake
        fake.eval.return_value = fake
        fake.return_value = torch.randn(1, 7)
        mock_torch_load.return_value = fake

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_path = "/tmp/model.pth"
        loader.device = torch.device("cpu")
        loader.class_names = list(range(7))

        with patch.object(loader, "_warmup_model", return_value=None):
            loader._load_model()

        assert fake.to.called
        assert fake.eval.called
