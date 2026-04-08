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

wait
