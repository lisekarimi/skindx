# 🚀 Deployment

This guide covers how to deploy SkinDX using Docker containers, including both Docker Compose for development and standalone Docker deployment with Nginx reverse proxy for production-like environments.

## 📦 Requirements
- Docker Desktop (with WSL2 on Windows recommended)
- Docker Compose (included in Docker Desktop)

## 🐳 Docker Compose

We use **Docker Compose** to manage multiple services in one stack:
- 📓 **Notebooks** → Jupyter Lab + MLflow (for experiments & tracking)
- 🌐 **App** → FastAPI (backend inference) + Streamlit (frontend UI)

This ensures all components run consistently, share resources (e.g. GPUs), and can be started together with one command — or separately if needed.

Run **everything together**:

```bash
make up
```

Or run services **individually**:

* `make notebooks` → Start **Jupyter + MLflow** (ports `8888`, `5000`)
* `make app` → Start **FastAPI + Streamlit** (ports `8000`, `7860`)

Other useful commands:

* `make build-all` → Build all services
* `make down` → Stop all running services

ℹ️ More useful commands are available in the **Makefile** inside the repository.


## 🐳 Docker Container

Besides Docker Compose, you can also run the app directly in a **standalone Docker container** with Nginx as a reverse proxy.

Makefile commands:
- `make build` → Build the app image
- `make run` → Run the app locally in Docker (with hot reload for development)
  - **Single entry point**: Port `80` (Nginx routes to all services)
  - Access main app at `http://localhost/`
  - Access docs at `http://localhost/docs`
  - Access API docs at `http://localhost/api-docs`
  - Check health at `http://localhost/health`

**Note:** Internally, Streamlit runs on port 7860 and FastAPI on port 8000, but Nginx routes all traffic through port 80, so you only need to access `http://localhost/`.

This is useful if you want to test the app **outside of Docker Compose**, in the same way it runs on PROD (GCP, AWS, HF space, ...).
