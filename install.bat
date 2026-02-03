@echo off
setlocal enabledelayedexpansion

echo ============================================
echo ArenaMCP v0.2.0 Installer
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: Activate virtual environment and install dependencies
echo Installing dependencies...
echo This may take several minutes on first install...
echo.

call venv\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip >nul 2>&1

:: Install the package with all dependencies
pip install -e ".[full]"
if errorlevel 1 (
    echo.
    echo WARNING: Some optional dependencies failed to install
    echo Core functionality should still work
    echo.
    :: Try installing just the base package
    pip install -e .
)

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo To run the coach:
echo   run.bat --backend gemini
echo   run.bat --backend ollama --model llama3.2
echo.
echo To run draft helper (no API key needed):
echo   run.bat --draft --set MH3
echo.
echo Hotkeys during gameplay:
echo   F4  = Push-to-talk (ask questions)
echo   F5  = Toggle mute
echo   F6  = Change voice
echo   F7  = Save bug report
echo   F8  = Swap seat (fix wrong player detection)
echo   F9  = Restart coach
echo.
echo Don't forget to set your API key in .env file!
echo.
pause
