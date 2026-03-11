@echo off
:: MTGA Autopilot Launcher
:: AI plays the game via mouse clicks

title MTGA Autopilot

:: Change to script directory (handles running from shortcut)
cd /d "%~dp0"

:: Prefer the venv Python so editable-installed packages are visible
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    :: Fall back to system Python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found in PATH
        echo Install from https://python.org
        pause
        exit /b 1
    )
    set "PY=python"
)

:: Run the launcher with --autopilot flag
%PY% launcher.py --autopilot %*

pause
