@echo off
title StressDetector Real-time Web Demo
echo =========================================================
echo       KHOI CHAY HE THONG CANDIDATE STRESS DETECTOR
echo =========================================================
echo.

:: 1. Kich hoat moi truong ao neu co
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Dang kich hoat venv tai .venv...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Dang kich hoat venv tai venv...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Khong tim thay moi truong ao. He thong se dung Python mac dinh cua Windows.
)

:: 2. Tu dong mo trinh duyet sau 3 giay (dam bao API da start kip)
echo [INFO] Dang chuan bi khoi chay trinh duyet...
start "" "http://localhost:8000"

:: 3. Chay Backend API bang Uvicorn
echo [INFO] Dang khoi dong FastAPI Uvicorn Server...
python -m uvicorn stress_benchmark.app_api:app --host 0.0.0.0 --port 8000 --log-level debug
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Co loi xay ra khi chay API Server!
    echo Vui long kiem tra xem ban da tai du thu vien bang lenh: pip install -r requirements.txt
    pause
)
