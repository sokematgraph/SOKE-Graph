@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM SOKEGraph - Automated Setup and Launch Script (Windows)
REM ═══════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM Configuration
set ENV_NAME=sokegraph
set PYTHON_VERSION=3.10

echo.
echo ═══════════════════════════════════════════════════════════════
echo    SOKEGraph - Research Paper Knowledge Graph Builder
echo ═══════════════════════════════════════════════════════════════
echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 1: Check if Conda is installed
REM ─────────────────────────────────────────────────────────────────
echo [1/4] Checking for Conda installation...

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Conda is not installed or not in PATH
    echo.
    echo Please install Miniconda or Anaconda from:
    echo   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   - Anaconda:  https://www.anaconda.com/download
    echo.
    echo After installation, restart Command Prompt and run this script again.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%i in ('conda --version') do set CONDA_VERSION=%%i
echo [OK] Conda found: %CONDA_VERSION%

REM ─────────────────────────────────────────────────────────────────
REM Step 2: Create/activate conda environment
REM ─────────────────────────────────────────────────────────────────
echo [2/4] Setting up conda environment '%ENV_NAME%'...

REM Check if environment exists
conda env list | findstr /r /c:"^%ENV_NAME% " >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Environment '%ENV_NAME%' already exists
) else (
    echo Creating new conda environment with Python %PYTHON_VERSION%...
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create conda environment
        pause
        exit /b 1
    )
    echo [OK] Environment created successfully
)

REM Activate the environment
echo Activating environment...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment
    pause
    exit /b 1
)

echo [OK] Environment '%ENV_NAME%' activated

REM ─────────────────────────────────────────────────────────────────
REM Step 3: Install dependencies
REM ─────────────────────────────────────────────────────────────────
echo [3/4] Installing dependencies from requirements.txt...

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in current directory
    pause
    exit /b 1
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed successfully

REM ─────────────────────────────────────────────────────────────────
REM Step 4: Launch Streamlit application
REM ─────────────────────────────────────────────────────────────────
echo [4/4] Starting SOKEGraph Streamlit application...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   SOKEGraph is starting...
echo   The application will open in your default browser
echo   Default URL: http://localhost:8501
echo.
echo   Press Ctrl+C to stop the application
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Run streamlit
streamlit run streamlit-app.py

REM This line will execute when the user stops streamlit
echo.
echo SOKEGraph application stopped.
pause
