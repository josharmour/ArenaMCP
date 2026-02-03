@echo off
setlocal

echo ArenaMCP v0.2.0
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

:: Activate venv and run
call venv\Scripts\activate.bat

:: Run standalone coach with all arguments passed through
python -m arenamcp.standalone %*
