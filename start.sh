#!/bin/bash

# Start FastAPI in background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in background
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 7860 &

# Wait for services to fully start
sleep 8

# Start nginx in foreground
nginx -g "daemon off;"
