# 🏗️ Architecture

SkinDX follows a modular architecture designed for scalability and maintainability.

## 🔍 System Overview

SKINDX has three main components:

- **Frontend (Streamlit)** → Web UI for uploading skin images and showing results.
- **Backend (FastAPI)** → Runs the ResNet-50 model and returns predictions.
- **Notebooks (Jupyter + MLflow)** → Used for dataset exploration, training, and evaluation.

All services run together with **Docker Compose**, ensuring consistent environments.

![Architecture](https://github.com/lisekarimi/skindx/blob/main/assets/static/arch.png?raw=true)


## 🧩 Core Components

### 🖥️ Frontend
- **Web UI**: Streamlit-based interface for user interaction
- **File Upload**: Support for image upload and validation
- **Results Display**: Visualization of classification results

### ⚙️ Backend Services
- **Model Service**: Handles model loading and inference
- **API Layer**: RESTful endpoints for external integration
- **File Processing**: Image preprocessing and validation

### 💾 Data Layer
- **Model Storage**: Trained model artifacts
- **Configuration**: Application settings and parameters

## 🛠️ Technology Stack

- **Python**: Core application language
- **Streamlit**: Web interface framework
- **PyTorch**: Deep learning framework
- **MLflow**: Model management and tracking
- **Docker**: Containerization
- **FastAPI**: API framework (if applicable)

## 🔄 Data Flow

1. User uploads image through web interface
2. Image is preprocessed and validated
3. Model performs inference on processed image
4. Results are formatted and returned to user
5. Classification results are displayed

### 🔀 Reverse Proxy Layer (Nginx)

SKINDX uses **Nginx** as a reverse proxy to provide a unified entry point for all services. This architecture allows multiple backend services (FastAPI on port 8000 and Streamlit on port 7860) to be accessed through a single port (80 in development, 8080 in production).

**Key benefits:**
- **Single Entry Point**: All services accessible through one domain/port
- **Request Routing**: Intelligently routes traffic based on URL paths
  - `/` → Streamlit UI (main application)
  - `/health`, `/predict` → FastAPI endpoints
  - `/docs` → Docsify documentation
  - `/api-docs` → Swagger API reference
- **WebSocket Support**: Handles Streamlit's real-time WebSocket connections
- **Load Balancing**: Can distribute traffic across multiple backend instances
- **Security**: Acts as a buffer between external requests and internal services

This setup ensures clean URLs in production (e.g., `https://skindx.lisekarimi.com/docs`) without exposing internal port numbers to end users.

## 📈 Scalability Considerations

- Stateless design for horizontal scaling
- Model caching for performance
- Asynchronous processing capabilities
- Container-based deployment
