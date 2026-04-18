@echo off
setlocal

cd /d "%~dp0"

start "FilingDelta Backend" cmd /k call "%~dp0start_backend.cmd"
start "FilingDelta Frontend" cmd /k call "%~dp0start_frontend.cmd"

echo FilingDelta launch commands have been sent.
echo Browser will open automatically when the frontend is ready.
echo Frontend: http://127.0.0.1:5173
echo Backend: http://127.0.0.1:8000
