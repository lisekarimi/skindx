FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -u 1000 appuser

# Configure environment
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/app
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV HF_HOME=/tmp/.cache
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy only pyproject.toml (NO uv.lock)
COPY pyproject.toml ./

# Create venv
RUN uv venv "$UV_PROJECT_ENVIRONMENT"

# Install PyTorch CPU-only FIRST
RUN uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies WITHOUT lock file
RUN uv pip install \
    requests \
    pillow \
    numpy \
    plotly \
    huggingface_hub \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    streamlit

# Copy application code
COPY . .

# Copy nginx configuration
COPY nginx.conf /etc/nginx/sites-available/default

# Set permissions
RUN chown -R appuser:appuser /app /opt/venv && \
    chown -R appuser:appuser /var/log/nginx /var/lib/nginx && \
    touch /var/run/nginx.pid && \
    chown appuser:appuser /var/run/nginx.pid

# Copy and fix line endings BEFORE switching user
COPY start.sh /app/start.sh
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh && chown appuser:appuser /app/start.sh

USER appuser

EXPOSE 80

CMD ["/app/start.sh"]
