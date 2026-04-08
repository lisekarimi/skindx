# 🔧 Services

Once the application is running, you can access the following interfaces:

## 🏠 Local Development

* 🏠 **Main App** → `http://localhost/`
* 📚 **Documentation** → `http://localhost/docs`
* ⚙️ **API Reference (Swagger)** → `http://localhost/api-docs`
* 💚 **Health Check** → `http://localhost/health`

## 🌐 Production

* 🏠 **Main App** → [https://skindx.lisekarimi.com/](https://skindx.lisekarimi.com/)
* 📚 **Documentation** → [https://skindx.lisekarimi.com/docs](https://skindx.lisekarimi.com/docs)
* ⚙️ **API Reference (Swagger)** → [https://skindx.lisekarimi.com/api-docs](https://skindx.lisekarimi.com/api-docs)
* 💚 **Health Check** → [https://skindx.lisekarimi.com/health](https://skindx.lisekarimi.com/health)

## 🛠️ Development Tools (Docker Compose only)

* 📓 **Jupyter Notebooks** → `http://localhost:8888`
* 📊 **MLflow Tracking** → `http://localhost:5000`

> **Note:** The first time you start notebooks, Jupyter requires an **access token**.
> Copy the full URL with `?token=...` from the container logs (shown in your terminal).
> Inside this environment, you can access **data exploration notebooks** and **MLflow** for experiment tracking.

---

**Architecture Note:** Internally, Streamlit runs on port 7860 and FastAPI on port 8000, but Nginx acts as a reverse proxy and routes all traffic through a single port (80 for local, 8080 for GCP Cloud Run).
