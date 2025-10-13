#!/bin/bash

# Start FastAPI in background
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Streamlit in background
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 7860 &
STREAMLIT_PID=$!

# Function to check if service is ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=120  # 10 minutes max (120 * 5 seconds)
    local attempt=0

    echo "Waiting for $service_name to be ready on port $port..."

    while [ $attempt -lt $max_attempts ]; do
        if [ "$port" = "8000" ]; then
            # For FastAPI, check /health endpoint and verify it returns 200 OK
            response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
            if [ "$response" = "200" ]; then
                echo "$service_name is ready with model loaded!"
                return 0
            else
                echo "Attempt $attempt/$max_attempts - FastAPI responding with HTTP $response (waiting for 200)..."
            fi
        else
            # For Streamlit, just check if port is responding
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

# Wait for FastAPI to be ready (with model loaded)
if ! wait_for_service 8000 "FastAPI"; then
    echo "FastAPI failed to start. Exiting..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
    exit 1
fi

# Wait for Streamlit to be ready
if ! wait_for_service 7860 "Streamlit"; then
    echo "Streamlit failed to start. Exiting..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
    exit 1
fi

# Start nginx in foreground
echo "All services ready! Starting nginx..."
nginx -g "daemon off;"
