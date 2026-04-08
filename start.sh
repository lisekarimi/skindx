#!/bin/bash
set -e

cleanup() {
    echo "Shutting down services..."
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    wait "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start FastAPI in background
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Streamlit in background
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
            if [ "$response" = "200" ]; then
                echo "$service_name is ready!"
                return 0
            fi
        else
            if curl -sf http://localhost:$port > /dev/null 2>&1; then
                echo "$service_name is ready!"
                return 0
            fi
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    echo "ERROR: $service_name failed to start within timeout"
    return 1
}

if ! wait_for_service 8000 "FastAPI"; then
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 1
fi

if ! wait_for_service 7860 "Streamlit"; then
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 1
fi

echo "All services ready!"
wait "$FASTAPI_PID" "$STREAMLIT_PID"
