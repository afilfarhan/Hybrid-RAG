@echo off
echo ========================================
echo Hybrid RAG - Setup Assistant
echo ========================================
echo.

echo [1/4] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found or not in PATH
    pause
    exit /b 1
)
echo.

echo [2/4] Installing dependencies...
echo This may take a few minutes...
echo.

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may not have installed correctly
    echo Continuing anyway...
    echo.
)

echo [3/4] Installing PDF support (recommended)...
pip install unstructured[pdf] pdfminer.six pdfplumber
if %errorlevel% neq 0 (
    echo WARNING: PDF support may not be fully installed
    echo You can still use text files
    echo.
)

echo.
echo [4/4] Creating directories...
if not exist "data\vector_store" mkdir data\vector_store
if not exist "data\sample_docs" mkdir data\sample_docs
if not exist "ui" mkdir ui

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Add your PDF files to: data\sample_docs\
echo.
echo 2. Run the ingestion script:
echo    python ingest_documents.py
echo.
echo 3. Start the server:
echo    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
echo.
echo 4. Visit: http://localhost:8000
echo.
echo ========================================
echo.

pause
