@echo off
:: MTGA Coach Launcher
:: Point your desktop shortcut to this file

title MTGA Coach

:: Change to script directory (handles running from shortcut)
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Install from https://python.org
    pause
    exit /b 1
)

:: Run the launcher (handles restarts)
python launcher.py %*

pause
