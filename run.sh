#!/bin/bash

echo "[INFO] Setting working directory to script location..."
cd "$(dirname "$0")" || exit 1

VENV_DIR="./venv"
PYTHON_VERSION="3.11"

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment found. Activating..."
    source "$VENV_DIR/bin/activate"
else
    echo "[INFO] No virtual environment found."
    
    if command -v py -$PYTHON_VERSION &> /dev/null; then
        echo "[INFO] Python $PYTHON_VERSION found. Creating virtual environment..."
        py -$PYTHON_VERSION -m venv venv
        source "$VENV_DIR/Scripts/activate"
        echo "[INFO] Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "[ERROR] Python $PYTHON_VERSION not found. Please install it first."
        exit 1
    fi
fi

echo "[INFO] Starting the backend server..."
python backend/backend.py