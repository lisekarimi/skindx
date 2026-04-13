# 📝 Changelog

## [0.3.0]

### 🔄 Changed
- Migrated deployment from GCP Cloud Run to Hugging Face Spaces (Docker)
- Removed Nginx — Streamlit serves directly on port 7860, FastAPI internal on 8000
- Docs moved to GitHub Pages with refreshed landing page
- Removed PageBotAI widget and FastAPI `/docs` static mount
- Updated landing page (`docs/index.html`): mobile-friendly navbar with hamburger menu, orange/skin theme, HF Spaces demo link
- Updated `Makefile` default port from `8080` to `7860`

### 🗑️ Removed
- `nginx.conf` — no longer needed


## [0.2.0]

### ✨ Added
- Nginx reverse proxy for unified service routing
- Docsify documentation at `/docs` endpoint
- Hot reload support for development (assets, src, docs folders)
- Production-ready Docker configuration

### 🔄 Changed
- FastAPI Swagger UI moved from `/docs` to `/api-docs`
- Disabled ReDoc documentation
- Single port access (80) for all services via Nginx
- Updated deployment structure with clean URL routing

### 🐛 Fixed
- Service access through unified entry point
- Mobile responsiveness for floating chat button
- Dark/light mode theming for Streamlit UI


## [0.1.0]
### ✨ Added
- Initial project structure (`src/`, `main.py`, `notebooks/`, `tests/`)
- FastAPI backend for model inference
- Streamlit frontend for image upload & analysis
- Model loader with Hugging Face integration (ResNet-50, HAM10000)
- Dockerfile & Docker Compose setup
- Makefile with build/run shortcuts
- CI/CD pipelines (linting, tests, security scan, deployment to HF Spaces)
- Training & evaluation notebooks with MLflow tracking
- Basic pytest test suite
