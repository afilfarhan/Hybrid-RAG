@echo off
echo ========================================
echo Hybrid RAG - Quick Start Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
echo.

REM Check if requirements are installed
echo [2/4] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may not have installed correctly
)
echo.

REM Create .env if it doesn't exist
if not exist .env (
    echo [3/4] Creating .env file...
    echo OPENAI_API_KEY=your_api_key_here > .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=8000 >> .env
    echo LOG_LEVEL=INFO >> .env
    echo.
    echo Please edit .env and add your OpenAI API key (optional)
    echo.
)

echo [4/4] Starting Hybrid RAG server...
echo.
echo Server will start on: http://localhost:8000
echo Chat interface: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
