@echo off
echo ==========================================
echo   AI Character Chat System - Starting...
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python 3.9+ first.
    pause
    exit /b 1
)


REM Check API Key
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo [Notice] GEMINI_API_KEY environment variable not detected.
    echo Please enter your Gemini API Key in config.py
    echo Or run: set GEMINI_API_KEY=your_key
    echo.
)

REM Start
echo Starting server...
echo.
echo Please open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python main.py
pause