#!/bin/bash
# run_server.sh

# Activate the virtual environment
source /home/pi/Projects/camera-app-fish/venv/bin/activate

cd /home/pi/Projects/camera-app-fish/

# Run FastAPI dev
uvicorn main:app --host 0.0.0.0 --port 8000
