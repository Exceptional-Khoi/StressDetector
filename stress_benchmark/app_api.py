import io
import logging
import math
import os
from importlib import metadata
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# --- ĐOẠN VÁ LỖI CHO NUMPY 2.X CHỐNG LỖI TRAPZ ---
if not hasattr(np, "trapz"):
    if hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    else:
        try:
            import scipy.integrate
            if hasattr(scipy.integrate, "trapz"):
                np.trapz = scipy.integrate.trapz
            elif hasattr(scipy.integrate, "trapezoid"):
                np.trapz = scipy.integrate.trapezoid
        except ImportError:
            def _fallback_trapz(y, x=None, dx=1.0, axis=-1):
                y = np.asanyarray(y)
                if x is None:
                    d = dx
                else:
                    x = np.asanyarray(x)
                    d = np.diff(x, axis=axis)
                slice1 = [slice(None)] * y.ndim
                slice2 = [slice(None)] * y.ndim
                slice1[axis] = slice(1, None)
                slice2[axis] = slice(None, -1)
                return np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)
            np.trapz = _fallback_trapz

# Import các file xử lý logic lõi từ hệ thống của bạn
from stress_benchmark.deploy import load_bundle, predict_feature_frame
from stress_benchmark.baseline import add_personal_baseline_features
from stress_benchmark.e4 import E4Record, extract_e4_features

# --- CONFIG ĐƯỜNG DẪN CHUẨN XÁC TRÊN WINDOWS ---
CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_FILE_DIR.parent

MODEL_PATH = Path(
    os.getenv(
        "STRESS_MODEL_PATH",
        PROJECT_ROOT / "outputs_scientific" / "models" / "binary_best_model.joblib",
    )
)

STATIC_DIR = CURRENT_FILE_DIR / "static"
BASELINE_MINUTES = 10
WINDOW_SEC = 60
STEP_SEC = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stress Detection API", version="1.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_BUNDLE = None

def _check_runtime_compatibility(bundle):
    trained = (bundle.get("dependency_versions") or {}).get("scikit-learn")
    try:
        runtime = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        runtime = None
    if trained and runtime and trained != runtime:
        raise RuntimeError(
            f"Model was trained with scikit-learn=={trained}, but server uses scikit-learn=={runtime}."
        )

@app.on_event("startup")
def startup():
    global MODEL_BUNDLE
    try:
        MODEL_BUNDLE = load_bundle(MODEL_PATH)
        _check_runtime_compatibility(MODEL_BUNDLE)
        logger.info("Model loaded successfully from %s.", MODEL_PATH)
    except Exception as e:
        MODEL_BUNDLE = None
        logger.error(f"Cannot load model: {e}")

def _needs_baseline(df: pd.DataFrame) -> bool:
    return not any(col.endswith("_delta_base") for col in df.columns)

def _sanitize(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val

def _parse_raw_timeseries_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm cầu nối: Đọc dữ liệu từ file CSV thô, trích xuất toàn bộ các luồng cảm biến
    bao gồm cả HR, bọc vào E4Record rồi gọi e4.py và features.py tính toán 623 cột.
    """
    subject_id = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else "unknown"
    session_id = str(df["session_id"].iloc[0]) if "session_id" in df.columns else "unknown"
    start_ts_utc = 0.0 

    signals = {}
    
    # 1. ACC (Gia tốc kế)
    if "acc_x" in df.columns and "acc_fs_hz" in df.columns:
        acc_df = df[df["acc_sample"] == 1][["acc_x", "acc_y", "acc_z"]].dropna()
        acc_fs = float(df["acc_fs_hz"].iloc[0])
        signals["acc"] = (acc_df.to_numpy(dtype=float), acc_fs)
        
    # 2. BVP (Sóng mạch thể tích)
    if "bvp" in df.columns and "bvp_fs_hz" in df.columns:
        bvp_arr = df[df["bvp_sample"] == 1]["bvp"].dropna().to_numpy(dtype=float)
        bvp_fs = float(df["bvp_fs_hz"].iloc[0])
        signals["bvp"] = (bvp_arr.reshape(-1, 1), bvp_fs)

    # 3. EDA (Điện trở da)
    if "eda" in df.columns and "eda_fs_hz" in df.columns:
        eda_arr = df[df["eda_sample"] == 1]["eda"].dropna().to_numpy(dtype=float)
        eda_fs = float(df["eda_fs_hz"].iloc[0])
        signals["eda"] = (eda_arr.reshape(-1, 1), eda_fs)

    # 4. TEMP (Nhiệt độ)
    if "temp" in df.columns and "temp_fs_hz" in df.columns:
        temp_arr = df[df["temp_sample"] == 1]["temp"].dropna().to_numpy(dtype=float)
        temp_fs = float(df["temp_fs_hz"].iloc[0])
        signals["temp"] = (temp_arr.reshape(-1, 1), temp_fs)

    # 5. HR (Nhịp tim thô - GIẢI PHÁP SỬA LỖI 500)
    if "hr" in df.columns and "hr_fs_hz" in df.columns:
        hr_arr = df[df["hr_sample"] == 1]["hr"].dropna().to_numpy(dtype=float)
        hr_fs = float(df["hr_fs_hz"].iloc[0])
        signals["hr"] = (hr_arr.reshape(-1, 1), hr_fs)

    # 6. IBI (Khoảng cách nhịp tim)
    if "ibi" in df.columns and "time_sec" in df.columns:
        ibi_df = df[df["ibi_event"] == 1][["time_sec", "ibi"]].dropna()
        signals["ibi_events"] = (ibi_df.to_numpy(dtype=float), -1.0)

    if not signals:
        raise ValueError("No valid sensor signals could be extracted from the CSV.")

    record = E4Record(
        subject_id=subject_id,
        session_id=session_id,
        start_ts_utc=start_ts_utc,
        signals=signals
    )

    # Gọi file e4.py, từ đó e4.py tự động gọi tiếp sang features.py để tính ra đủ cột
    return extract_e4_features(record, window_sec=WINDOW_SEC, step_sec=STEP_SEC)


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
        
    hardcoded_path = Path(r"D:\StressDetector-main\stress_benchmark\static\index.html")
    if hardcoded_path.is_file():
        return FileResponse(str(hardcoded_path))
        
    logger.error("Không tìm thấy file index.html!")
    raise HTTPException(status_code=404, detail="index.html not found.")


@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    content = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Bước 1: Gọi các file xử lý liên kết bên ngoài để tính toán ra dữ liệu đầy đủ cột
    try:
        features_df = _parse_raw_timeseries_to_features(raw_df)
    except Exception as e:
        logger.exception("Feature extraction failed")
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {e}")

    # Bước 2: Gọi file baseline.py để chuẩn hóa
    if _needs_baseline(features_df):
        try:
            features_df = add_personal_baseline_features(features_df, baseline_minutes=BASELINE_MINUTES)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Baseline feature error: {e}")

    # Bước 3: Đưa dữ liệu hoàn chỉnh vào file deploy.py để đưa vào models chạy dự đoán
    try:
        results_df = predict_feature_frame(MODEL_BUNDLE, features_df)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Bước 4: Chuẩn hóa kết quả trả về cấu trúc cho frontend nhận dạng
    if "decision_adjusted_label" in results_df.columns:
        results_df["adjusted_label"] = results_df["decision_adjusted_label"]
    elif "adjusted_label" not in results_df.columns:
        results_df["adjusted_label"] = 0

    for col in ["alert_state", "recommendation"]:
        if col not in results_df.columns:
            results_df[col] = "N/A"
    if "confidence" not in results_df.columns:
        results_df["confidence"] = 0.0

    output_cols = [c for c in [
        "window_start_sec", "window_start_ts",
        "pred_label", "adjusted_label", "confidence",
        "alert_state", "recommendation",
    ] if c in results_df.columns]
    output_cols += [c for c in results_df.columns if c.startswith("proba_")]

    records = results_df[output_cols].to_dict(orient="records")
    records = [{k: _sanitize(v) for k, v in row.items()} for row in records]
    return records
