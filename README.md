# 🔬 SKINDX – AI Skin Lesion Analysis

Upload a skin photo and the AI will classify it into one of 7 lesion types.

[🚀 **Try the Live Demo**](https://skindx.lisekarimi.com)

<img src="https://github.com/lisekarimi/skindx/blob/main/assets/static/fullpage.png?raw=true" alt="SkindDx interface" width="450">

**SkinDx is an AI-powered web app** that uses a fine-tuned **ResNet-50** CNN model trained on the **HAM10000 dataset** to classify skin lesions into 7 categories.

## 🚀 Features

* Upload skin lesion images → get predicted class and confidence.
* Supports **7 skin lesion types** (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
* Confidence breakdown visualization with urgency indicators.

## 🔬 How SkinDx Works

1. 📸 Upload a skin lesion photo
2. 🔍 Click **Analyze**
3. 🤖 AI model (ResNet-50) predicts lesion type + confidence
4. 📊 Results shown in the app (class, risk, confidence, chart)

⚠️ **Disclaimer:** This tool is **not a medical device**.
Results are for **educational purposes only** – always consult a dermatologist.

## 🛠️ Pre-requisites

- Python 3.11.x (not 3.12+)
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Make: `winget install GnuWin32.Make` (Windows) | `brew install make` (macOS) | `sudo apt install make` (Linux)
- GPU – model trained on RTX 5080

## 🔧 Setup & Architecture

Runs with **Docker Desktop** (WSL2 on Windows recommended). This ensures consistent environments across systems.

Components:
- **Streamlit** → Web UI
- **FastAPI** → Model inference API
- **Jupyter/MLflow** → Notebooks & experiment tracking

Everything runs together via **Docker Compose** (already included in Docker Desktop).

Start with:

```bash
docker compose up --build
```

Or, if you prefer **Makefile** shortcuts:

```bash
make up
```

* Streamlit UI → `http://localhost:7860`
* FastAPI Docs (**Swagger UI**) → `http://localhost:8000/api-docs`
* Jupyter/MLflow (notebooks) → `http://localhost:8888`
  > ⚠️ The first time you start notebooks, Jupyter requires an **access token**.
  > Copy the full URL with `?token=...` from the container logs (shown in the terminal).


## 📂 Project Structure

* `main.py` → FastAPI inference service
* `src/ui/` → Streamlit frontend
* `src/model_loader.py` → HuggingFace model loader
* `notebooks/` → Data exploration, training & evaluation
* `tests/` → Pytest unit/integration tests
* Model hosted on HF: **`https://huggingface.co/lisekarimi/resnet50-ham10000`**


## 📊 Training Pipeline (in notebooks)

1. **Data exploration** (HAM10000 distribution)
2. **Class imbalance handling** (smart augmentation)
3. **ResNet-50 training & tracking with MLflow**
4. **External validation** on ISIC 2019 dataset

## ⚙️ CI/CD

This project includes a full **CI/CD pipeline** (tests, linting, security scans, deployment in Hugging Face).


## 📖 Documentation

Additional details are available [here](https://skindx.lisekarimi.com/docs/).
