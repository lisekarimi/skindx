#!/bin/bash

FASTAPI_PID=""

cleanup() {
    echo "Shutting down..."
    [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

wait_for_service() {
    local max_attempts=120
    local attempt=0

    echo "Waiting for FastAPI on port 8000..."
    while [ $attempt -lt $max_attempts ]; do
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
        [ "$response" = "200" ] && echo "FastAPI is ready!" && return 0
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "ERROR: FastAPI failed to start"
    return 1
}

if ! wait_for_service; then cleanup; fi

echo "Starting Streamlit on port 7860..."
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 7860
