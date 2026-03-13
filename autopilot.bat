@echo off
setlocal

:: Autopilot wrapper (keeps existing shortcut target stable)
cd /d "%~dp0"
call launch.bat --autopilot %*
exit /b %errorlevel%
