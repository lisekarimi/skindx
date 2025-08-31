# src/model_loader.py
import logging
import time

import requests
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from src.constants import CLASS_NAMES, HF_MODEL_FILENAME, HF_REPO_ID
from src.utils import preprocess_image

logger = logging.getLogger(__name__)

HF_REVISION = "217a9639ec46e2c5fd241973433c6ad69f984f54"


class ModelLoader:
    """A class to load and manage a trained PyTorch model for making predictions.

    This class handles model loading, device management, and inference operations
    for image classification tasks.

    Attributes
    ----------
        model_path (str): Path to the saved model file.
        device (torch.device): Device to run the model on (CPU or CUDA).
        model: The loaded PyTorch model.
        class_names (list): List of class names for classification.

    """

    def __init__(self):
        """Initialize the ModelLoader by downloading model from Hugging Face."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = list(CLASS_NAMES.keys())

        logger.info(f"Using device: {self.device}")

        # Download and load model from HF with retry logic
        try:
            logger.info(f"Downloading model from Hugging Face: {HF_REPO_ID}")
            self.model_path = self._download_with_retry(HF_REPO_ID, HF_MODEL_FILENAME)
            logger.info(f"Model downloaded to: {self.model_path}")
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face: {e}")
            raise e

    def _download_with_retry(
        self, repo_id: str, filename: str, max_retries: int = 3, delay: float = 2.0
    ) -> str:
        """Download model with retry logic for network failures."""
        for attempt in range(max_retries):
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision="217a9639ec46e2c5fd241973433c6ad69f984f54",
                )
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    raise ValueError(
                        "Authentication failed. Check HF_TOKEN environment variable"
                    ) from e
                elif e.response.status_code == 429:
                    # Rate limited - use longer delay
                    if attempt == max_retries - 1:
                        raise ValueError(
                            "Rate limited by HuggingFace. Try again later"
                        ) from e
                    logger.warning(f"Rate limited. Retrying in {delay * 5} seconds...")
                    time.sleep(delay * 5)
                    continue
                elif e.response.status_code >= 500:
                    # Server error - retry
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(
                        f"Server error {e.response.status_code}. Retrying..."
                    )
                    time.sleep(delay * (attempt + 1))
                else:
                    # Other HTTP errors - don't retry
                    raise e
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Network error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay * (attempt + 1))
            except Exception as e:
                # Handle repository not found and other general errors
                if "not found" in str(e).lower() or "access denied" in str(e).lower():
                    raise ValueError(
                        f"Repository {repo_id} not found or access denied"
                    ) from e
                # For other unknown errors, retry
                if attempt == max_retries - 1:
                    raise e
                logger.warning(
                    f"Download attempt {attempt + 1} failed: {e}. Retrying..."
                )
                time.sleep(delay * (attempt + 1))

        raise Exception("All download attempts failed")

    def _load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load model
            self.model = torch.load(  # nosec B614
                self.model_path, map_location=self.device, weights_only=False
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded and set to evaluation mode")

            # Test model with dummy input
            self._validate_model()
            # Prevent slow first requests caused by model initialization.
            self._warmup_model()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _validate_model(self):
        """Validate model has expected architecture and outputs."""
        try:
            # Test with dummy input to check output shape
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)

            # Validate output shape matches expected number of classes
            expected_classes = len(self.class_names)
            if output.shape[1] != expected_classes:
                raise ValueError(
                    f"Model output shape {output.shape} "
                    f"doesn't match expected classes {expected_classes}"
                )

            # Validate output is proper logits (not already softmaxed)
            if torch.any(output < 0):
                logger.info("Model outputs raw logits (expected)")
            else:
                logger.warning("Model might output probabilities instead of logits")

            logger.info(f"Model validation successful. Output shape: {output.shape}")

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise e

    def _warmup_model(self):
        """Warm up the model with multiple dummy predictions to optimize performance."""
        logger.info("Warming up model for optimal performance...")
        try:
            # Run multiple warmup predictions with different batch sizes
            dummy_inputs = [
                torch.randn(1, 3, 224, 224).to(self.device),
                torch.randn(2, 3, 224, 224).to(self.device),  # Batch of 2
            ]

            with torch.no_grad():
                for i, dummy_input in enumerate(dummy_inputs):
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    warmup_time = time.time() - start_time
                    logger.info(f"Warmup {i + 1} completed in {warmup_time:.3f}s")

            logger.info("Model warmup completed successfully")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}. First prediction may be slower.")

    def predict(self, image: Image.Image) -> dict:
        """Make prediction on a single image."""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Preprocess image
            input_tensor = preprocess_image(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                # Convert to CPU and numpy for JSON serialization
                probabilities = probabilities.cpu().numpy()[0]
                confidence = confidence.cpu().item()
                predicted_idx = predicted_idx.cpu().item()

                # Create results dictionary
                predicted_class = self.class_names[predicted_idx]

                all_probabilities = {
                    self.class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }

                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": all_probabilities,
                    "predicted_index": predicted_idx,
                }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e
