@echo off
setlocal

cd /d "%~dp0frontend"
title FilingDelta Frontend

if not exist "node_modules\" (
    echo [FilingDelta] Frontend dependencies are missing.
    echo Please run: cd frontend ^&^& npm.cmd install
    exit /b 1
)

npm.cmd run dev -- --host 127.0.0.1 --port 5173 --open
