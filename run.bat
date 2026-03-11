@echo off
setlocal

echo ArenaMCP v0.5.10
echo.

:: Change to script directory (handles running from shortcut)
cd /d "%~dp0"

:: Check if venv exists
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found in PATH
        echo Please run install.bat first
        pause
        exit /b 1
    )
    set "PY=python"
)

:: Run standalone coach with all arguments passed through
%PY% -m arenamcp.standalone %*
