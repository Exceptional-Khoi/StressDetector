@echo off
setlocal
cd /d "%~dp0"

rem Keep Python from importing packages from AppData\Roaming user-site.
set PYTHONNOUSERSITE=1
set PYTHONPATH=%CD%

"C:\Users\DELL\anaconda3\python.exe" -m uvicorn stress_benchmark.app_api:app --host 127.0.0.1 --port 8000
