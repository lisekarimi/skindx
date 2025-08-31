FROM python:3.11-slim

# Install uv package manager from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create a non-root user early
RUN useradd -m -u 1000 appuser

# Configure Python virtual environment location
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/app

# Use system temp directory for caches (writable in restricted environments)
ENV HF_HOME=/tmp/.cache
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit

# Configure Streamlit to run headless without user interaction
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies as root
RUN uv venv "$UV_PROJECT_ENVIRONMENT" && uv sync --no-dev --extra app

# Copy application source code
COPY . .

# Create writable directories and set proper ownership
RUN mkdir -p /tmp/.cache /tmp/.streamlit /app/uploads && \
    chown -R appuser:appuser /app /opt/venv /tmp/.cache /tmp/.streamlit

# Switch to non-root user
USER appuser

# Expose ports for both services
EXPOSE 7860 8000

CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 7860 --server.enableXsrfProtection=false"]
