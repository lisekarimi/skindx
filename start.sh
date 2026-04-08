#!/bin/bash

FASTAPI_PID=""
STREAMLIT_PID=""

cleanup() {
    echo "Shutting down..."
    [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null
    [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 7860 &
STREAMLIT_PID=$!

wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=120
    local attempt=0

    echo "Waiting for $service_name on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if [ "$port" = "8000" ]; then
            response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
            [ "$response" = "200" ] && echo "$service_name is ready!" && return 0
        else
            curl -sf http://localhost:$port > /dev/null 2>&1 && echo "$service_name is ready!" && return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "ERROR: $service_name failed to start"
    return 1
}

if ! wait_for_service 8000 "FastAPI"; then cleanup; fi
if ! wait_for_service 7860 "Streamlit"; then cleanup; fi

echo "All services ready! Starting nginx on port ${PORT}..."
PORT=${PORT:-8080} envsubst '${PORT}' < /app/nginx.conf > /tmp/nginx.conf
nginx -c /tmp/nginx.conf -g "daemon off;"
