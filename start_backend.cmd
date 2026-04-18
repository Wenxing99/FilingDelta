@echo off
setlocal

cd /d "%~dp0"
title FilingDelta Backend

uv run uvicorn filingdelta.main:app --reload --host 127.0.0.1 --port 8000
