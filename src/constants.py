# src/ui/constants.py

import tomllib
from pathlib import Path

# ==================== PROJECT METADATA ====================
root = Path(__file__).resolve().parent.parent
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

PROJECT_NAME = pyproject["project"]["name"]
VERSION = pyproject["project"]["version"]

# ==================== MODEL CONFIGURATION ====================
# Hugging Face model configuration
HF_REPO_ID = "lisekarimi/resnet50-ham10000"
HF_MODEL_FILENAME = "resnet50_v010.pth"

# URL for the local API server
API_URL = "http://localhost:8000"

# Model classes
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
