# main.py
import io
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.constants import (
    VERSION,
)
from src.model_loader import ModelLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SKINDX - Skin Lesion Classification API",
    description="AI-powered skin lesion classification using ResNet",
    version=VERSION,
    docs_url="/api-docs",  # Change from default /docs to /api-docs
    redoc_url=None,
)

# Mount static files for documentation
docs_path = Path(__file__).parent / "docs"
if docs_path.exists():
    app.mount("/docs", StaticFiles(directory=str(docs_path), html=True), name="docs")
    logger.info(f"Documentation mounted at /docs from {docs_path}")
else:
    logger.warning(f"Documentation directory not found at {docs_path}")

# Global model loader
model_loader = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_loader
    try:
        logger.info("Loading model from Hugging Face")
        model_loader = ModelLoader()  # No parameters needed
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    return {
        "message": "SKINDX API",
        "version": VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "api_docs": "/api-docs",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Return API health status and model load state."""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable - model not loaded yet"
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "version": VERSION,
    }


@app.post("/predict")
async def predict_lesion(file: UploadFile | None = None):
    """Predict skin lesion type from uploaded image."""
    file = File(...) if file is None else file
    if not model_loader:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Add file size validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {file.size} bytes exceeds maximum "
                f"allowed size of {MAX_FILE_SIZE} bytes"
            ),
        )

    try:
        start_time = time.time()

        # Read and preprocess image
        image_data = await file.read()

        # Validate actual content size if file.size was None
        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File content size exceeds {MAX_FILE_SIZE} bytes",
            )

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get prediction
        prediction_result = model_loader.predict(image)

        processing_time = time.time() - start_time

        return JSONResponse(
            content={
                "success": True,
                "predicted_class": prediction_result["predicted_class"],
                "confidence": float(prediction_result["confidence"]),
                "all_probabilities": {
                    k: float(v) for k, v in prediction_result["probabilities"].items()
                },
                "processing_time": round(processing_time, 4),
                "image_info": {"filename": file.filename, "size": image.size},
                "model_source": "huggingface",
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        ) from e
