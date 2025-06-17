@echo off
echo [INFO] Setting working directory to script location...
cd /d %~dp0

set "VENV_DIR=venv"
set "PYTHON_VERSION=3.11"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Virtual environment found. Activating...
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo [INFO] No virtual environment found.
    where python%PYTHON_VERSION% >nul 2>&1
    if %errorlevel%==0 (
        echo [INFO] Python %PYTHON_VERSION% found. Creating virtual environment...
        python%PYTHON_VERSION% -m venv venv
        call "%VENV_DIR%\Scripts\activate.bat"
        echo [INFO] Installing dependencies...
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    ) else (
        echo [ERROR] Python %PYTHON_VERSION% not found. Please install it first.
        pause
        exit /b 1
    )
)

echo [INFO] Starting the backend server...
python backend\backend.py
