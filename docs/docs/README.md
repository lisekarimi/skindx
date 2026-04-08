# 🔬 SkinDX - Skin Lesion Classification

SkinDX is designed to classify skin lesions into different categories using state-of-the-art computer vision models. The application provides both a web interface and API endpoints for skin lesion analysis.

## 🩺 How SkinDx Works

1. 📸 Upload a skin lesion photo
2. 🔍 Click **Analyze**
3. 🤖 AI model (ResNet-50) predicts lesion type + confidence
4. 📊 Results shown in the app (class, risk, confidence, chart)

## ✨ Key Features

- Deep learning model for skin lesion classification
- Web-based user interface
- RESTful API endpoints
- Model training pipeline
- Comprehensive testing suite
- CI/CD integration

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


## 📚 Documentation

- 🏗️ Architecture - System design and components
- 🐳 Setup & Deployment - Installation and deployment
- ⚙️ CI/CD - Continuous integration and deployment
- 🧪 Testing Strategy - Testing approach and guidelines
- 📊 Training Pipeline - Model training process
- 📦 Model Management - Model versioning and storage
- 📡 Services - Service architecture and APIs
- 🤝 Contributing - Development guidelines

## 🆘 Getting Help

For questions and support, you can:
- Connect with us on [LinkedIn](https://linkedin.com/in/lisekarimi/)
