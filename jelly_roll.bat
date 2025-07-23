@echo off
echo.
echo 🎐 Jellyfish Dynamite - Audio Analysis Tool 🎐
echo ================================================
echo.
echo Setting up your environment...
echo This may take 2-3 minutes depending on your internet connection.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Found Python %PYTHON_VERSION%

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install requirements
echo.
echo 📚 Installing required packages...
echo This is the longest step - please wait...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    echo.
    echo Try running this command manually:
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ✅ Installation complete!
echo.
echo 🚀 Starting Jellyfish Dynamite...
echo.
echo When you see "Running on http://127.0.0.1:5000", 
echo open your web browser and go to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server when you're done.
echo.

REM Start the Flask app
python jelly_app.py

echo.
echo 👋 Jellyfish Dynamite has stopped.
echo You can run this batch file again anytime to restart the application.
pause