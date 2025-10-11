FROM python:3.11-slim

# Install nginx and uv
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
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

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv venv "$UV_PROJECT_ENVIRONMENT" && uv sync --no-dev --extra app

# Copy source code
COPY . .

# Copy nginx configuration
COPY nginx.conf /etc/nginx/sites-available/default

# Set proper permissions
RUN chown -R appuser:appuser /app /opt/venv && \
    chown -R appuser:appuser /var/log/nginx /var/lib/nginx && \
    touch /var/run/nginx.pid && \
    chown appuser:appuser /var/run/nginx.pid

# Switch to non-root user
USER appuser

# Expose nginx port
EXPOSE 80

# Copy startup script
COPY --chown=appuser:appuser start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run nginx with both services
CMD ["/app/start.sh"]
