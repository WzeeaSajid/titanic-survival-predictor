#!/bin/bash
# Titanic Survival Predictor — EC2 Deployment Script
# Usage: chmod +x deploy.sh && ./deploy.sh

set -e

echo "=== Titanic Survival Predictor — Deployment ==="

# Detect Python
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Python 3 not found. Installing..."
    sudo yum update -y
    sudo yum install -y python3
    PYTHON_CMD="python3"
fi

echo "Using: $($PYTHON_CMD --version)"

# Install pip if missing
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Installing pip..."
    sudo yum install -y python3-pip
fi

# Install dependencies
echo "Installing dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Verify model exists
if [ ! -f "models/model.pkl" ]; then
    echo "ERROR: models/model.pkl not found. Train the model first:"
    echo "  $PYTHON_CMD -m src.train"
    exit 1
fi

echo "Model found: models/model.pkl"

# Start the API
echo "Starting API on port 8000..."
exec $PYTHON_CMD -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info
