from __future__ import annotations

import io
import pickle
import re
import zipfile
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baseline import add_personal_baseline_features
from .config import ExtractionConfig
from .e4 import extract_e4_features, read_e4_zip_bytes
from .features import window_features


WESAD_LABEL_FS = 700.0
WESAD_WRIST_FS = {
    "ACC": 32.0,
    "BVP": 64.0,
    "EDA": 4.0,
    "TEMP": 4.0,
}

WESAD_SIGNAL_MAP = {
    "ACC": ("acc", WESAD_WRIST_FS["ACC"]),
    "BVP": ("bvp", WESAD_WRIST_FS["BVP"]),
    "EDA": ("eda", WESAD_WRIST_FS["EDA"]),
    "TEMP": ("temp", WESAD_WRIST_FS["TEMP"]),
}


def normalize_subject_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def mode_and_purity(values: np.ndarray) -> Tuple[Optional[int], float]:
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0.0
    counts = Counter(arr.astype(int).tolist())
    label, count = counts.most_common(1)[0]
    return int(label), float(count / arr.size)


def _find_matching_offset_samples(raw_values: np.ndarray, pkl_values: np.ndarray) -> Optional[int]:
    raw = np.asarray(raw_values, dtype=float)
    pkl = np.asarray(pkl_values, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if pkl.ndim == 1:
        pkl = pkl.reshape(-1, 1)
    if raw.size == 0 or pkl.size == 0 or raw.shape[1] < pkl.shape[1]:
        return None

    n_cols = pkl.shape[1]
    n_match = min(256, len(pkl), len(raw))
    if n_match < 8:
        return None
    needle = pkl[:n_match, :n_cols]
    first = needle[0]
    candidate_mask = np.all(np.isclose(raw[:, :n_cols], first, rtol=0.0, atol=1e-10), axis=1)
    candidates = np.flatnonzero(candidate_mask)
    for start_idx in candidates:
        end_idx = int(start_idx) + n_match
        if end_idx > len(raw):
            continue
        if np.allclose(raw[int(start_idx) : end_idx, :n_cols], needle, rtol=0.0, atol=1e-10):
            return int(start_idx)
    return None


def _wesad_pkl_offset_sec(record_signals: Dict[str, Tuple[np.ndarray, float]], wrist: Dict[str, np.ndarray]) -> Optional[float]:
    offsets: Dict[str, float] = {}
    for pkl_name, (signal_name, fs) in WESAD_SIGNAL_MAP.items():
        if signal_name not in record_signals or pkl_name not in wrist:
            continue
        raw_values, raw_fs = record_signals[signal_name]
        if not np.isclose(float(raw_fs), fs):
            continue
        offset_samples = _find_matching_offset_samples(raw_values, np.asarray(wrist[pkl_name], dtype=float))
        if offset_samples is not None:
            offsets[signal_name] = float(offset_samples / raw_fs)
    # HR and IBI are derived from the E4 PPG stream, so BVP alignment is the
    # most appropriate reference when it is available.
    if "bvp" in offsets:
        return offsets["bvp"]
    if offsets:
        return float(np.median(list(offsets.values())))
    return None


def _crop_signal(values: np.ndarray, fs: float, start_sec: float, duration_sec: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    expected_len = max(1, int(round(duration_sec * fs)))
    start_idx = max(0, int(round(start_sec * fs)))
    end_idx = start_idx + expected_len
    if start_idx >= len(arr):
        pad_shape = (expected_len,) + arr.shape[1:]
        return np.full(pad_shape, np.nan)
    cropped = arr[start_idx : min(end_idx, len(arr))]
    if len(cropped) < expected_len:
        pad_shape = (expected_len - len(cropped),) + arr.shape[1:]
        cropped = np.concatenate([cropped, np.full(pad_shape, np.nan)], axis=0)
    return cropped


def _crop_events(values: np.ndarray, start_sec: float, duration_sec: float) -> np.ndarray:
    events = np.asarray(values, dtype=float)
    if events.ndim != 2 or events.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    end_sec = start_sec + duration_sec
    times = events[:, 0]
    mask = np.isfinite(times) & (times >= start_sec) & (times < end_sec)
    cropped = events[mask, :2].copy()
    if cropped.size:
        cropped[:, 0] = cropped[:, 0] - start_sec
    return cropped


def _read_wesad_e4_heart_signals(
    zf: zipfile.ZipFile,
    subject_id: str,
    wrist: Dict[str, np.ndarray],
    duration_sec: float,
) -> Dict[str, Tuple[np.ndarray, float]]:
    e4_name = f"WESAD/{subject_id}/{subject_id}_E4_Data.zip"
    try:
        record = read_e4_zip_bytes(subject_id, subject_id, zf.read(e4_name))
    except KeyError:
        return {}
    if record is None:
        return {}
    offset_sec = _wesad_pkl_offset_sec(record.signals, wrist)
    if offset_sec is None:
        return {}

    extra: Dict[str, Tuple[np.ndarray, float]] = {}
    for signal_name in ("hr", "ibi", "ibi_events"):
        if signal_name not in record.signals:
            continue
        values, fs = record.signals[signal_name]
        if signal_name == "ibi_events":
            extra[signal_name] = (_crop_events(values, offset_sec, duration_sec), float(fs))
        else:
            extra[signal_name] = (_crop_signal(values, float(fs), offset_sec, duration_sec), float(fs))
    return extra


def build_wesad_features(config: ExtractionConfig) -> pd.DataFrame:
    if not config.wesad_zip.exists():
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    with zipfile.ZipFile(config.wesad_zip) as zf:
        pkl_names = sorted(
            [name for name in zf.namelist() if re.match(r"WESAD/S\d+/S\d+\.pkl$", name)],
            key=lambda n: int(re.search(r"S(\d+)\.pkl$", n).group(1)),
        )
        if config.max_wesad_subjects:
            pkl_names = pkl_names[: config.max_wesad_subjects]
        for pkl_name in pkl_names:
            subject_id = re.search(r"/(S\d+)/", pkl_name).group(1)
            with zf.open(pkl_name) as fh:
                data = pickle.load(fh, encoding="latin1")
            labels = np.asarray(data["label"]).reshape(-1)
            wrist = data["signal"]["wrist"]
            signals = {
                "acc": (np.asarray(wrist["ACC"], dtype=float), WESAD_WRIST_FS["ACC"]),
                "bvp": (np.asarray(wrist["BVP"], dtype=float), WESAD_WRIST_FS["BVP"]),
                "eda": (np.asarray(wrist["EDA"], dtype=float), WESAD_WRIST_FS["EDA"]),
                "temp": (np.asarray(wrist["TEMP"], dtype=float), WESAD_WRIST_FS["TEMP"]),
            }
            duration = len(labels) / WESAD_LABEL_FS
            signals.update(_read_wesad_e4_heart_signals(zf, subject_id, wrist, duration))
            start = 0.0
            while start + config.window_sec <= duration + 1e-6:
                end = start + config.window_sec
                label_start = int(round(start * WESAD_LABEL_FS))
                label_end = int(round(end * WESAD_LABEL_FS))
                protocol_label, purity = mode_and_purity(labels[label_start:label_end])
                if protocol_label not in {1, 2, 3} or purity < config.label_purity:
                    start += config.step_sec
                    continue
                row = window_features(signals, start, end)
                row["source"] = "wesad"
                row["subject_id"] = subject_id
                row["session_id"] = subject_id
                row["group_id"] = f"wesad:{subject_id}"
                row["wesad_protocol_label"] = protocol_label
                row["label_purity"] = purity
                # Unified task labels: baseline/amusement are non-stress; TSST is high stress.
                row["target3"] = 2 if protocol_label == 2 else 0
                row["target_binary"] = 1 if protocol_label == 2 else 0
                row["window_start_ts"] = np.nan
                row["window_end_ts"] = np.nan
                rows.append(row)
                start += config.step_sec
    return pd.DataFrame(rows)


def _parse_excel_time(value: object) -> Optional[time]:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    text = str(value).strip()
    try:
        parts = [int(float(part)) for part in text.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return time(parts[0], parts[1], parts[2])
    except Exception:
        return None


def _parse_excel_date(value: object) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return datetime(parsed.year, parsed.month, parsed.day)


def load_survey_intervals(path: Path, survey_offset_hours: float) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="in")
    rows = []
    for _, row in raw.iterrows():
        subject_id = normalize_subject_id(row.get("ID"))
        if not subject_id:
            continue
        label = row.get("Stress level")
        if pd.isna(label) or str(label).strip().lower() == "na":
            continue
        try:
            label_int = int(float(label))
        except Exception:
            continue
        if label_int not in {0, 1, 2}:
            continue
        date_value = _parse_excel_date(row.get("date"))
        start_time = _parse_excel_time(row.get("Start time"))
        end_time = _parse_excel_time(row.get("End time"))
        if date_value is None or start_time is None or end_time is None:
            continue
        local_start = datetime.combine(date_value.date(), start_time)
        local_end = datetime.combine(date_value.date(), end_time)
        if local_end < local_start:
            local_end += timedelta(days=1)
        start_utc = (local_start - timedelta(hours=survey_offset_hours)).replace(tzinfo=timezone.utc)
        end_utc = (local_end - timedelta(hours=survey_offset_hours)).replace(tzinfo=timezone.utc)
        rows.append(
            {
                "subject_id": subject_id,
                "survey_start_ts": start_utc.timestamp(),
                "survey_end_ts": end_utc.timestamp(),
                "nurse_label": label_int,
            }
        )
    return pd.DataFrame(rows)


def _session_entries(nurse_zip: Path) -> List[zipfile.ZipInfo]:
    with zipfile.ZipFile(nurse_zip) as zf:
        entries = [
            info
            for info in zf.infolist()
            if (not info.is_dir()) and re.match(r"[^/]+/[^/]+_\d+\.zip$", info.filename)
        ]
    return sorted(entries, key=lambda info: info.filename)


def scan_survey_offsets(data_dir: Path, offsets: Sequence[float] = tuple(range(-12, 15))) -> pd.DataFrame:
    nurse_zip = data_dir / "Stress_dataset.zip"
    survey_path = data_dir / "SurveyResults.xlsx"
    entries = _session_entries(nurse_zip)
    session_rows = []
    for info in entries:
        match = re.match(r"([^/]+)/[^_]+_(\d+)\.zip$", info.filename)
        if not match:
            continue
        session_rows.append((match.group(1), float(match.group(2))))
    rows = []
    for offset in offsets:
        intervals = load_survey_intervals(survey_path, float(offset))
        by_subject = {sid: grp for sid, grp in intervals.groupby("subject_id")}
        matched = 0
        subjects = Counter()
        for subject_id, start_ts in session_rows:
            grp = by_subject.get(normalize_subject_id(subject_id))
            if grp is None:
                continue
            hit = grp[(grp["survey_start_ts"] <= start_ts) & (start_ts <= grp["survey_end_ts"])]
            if not hit.empty:
                matched += 1
                subjects[subject_id] += 1
        rows.append(
            {
                "survey_offset_hours": float(offset),
                "matched_session_starts": int(matched),
                "matched_subjects": int(len(subjects)),
            }
        )
    return pd.DataFrame(rows).sort_values(["matched_session_starts", "matched_subjects"], ascending=False)


def choose_best_survey_offset(data_dir: Path) -> float:
    scan = scan_survey_offsets(data_dir)
    if scan.empty:
        return 0.0
    return float(scan.iloc[0]["survey_offset_hours"])


def _label_nurse_windows(
    features: pd.DataFrame,
    intervals: pd.DataFrame,
    min_overlap: float,
    boundary_margin_sec: float = 0.0,
) -> pd.DataFrame:
    if features.empty or intervals.empty:
        return features
    out = features.copy()
    out["nurse_label"] = np.nan
    out["label_overlap_ratio"] = 0.0
    interval_groups = {sid: grp.sort_values("survey_start_ts") for sid, grp in intervals.groupby("subject_id")}
    labels = []
    overlaps = []
    for _, row in out.iterrows():
        subject_id = normalize_subject_id(row["subject_id"])
        grp = interval_groups.get(subject_id)
        if grp is None:
            labels.append(np.nan)
            overlaps.append(0.0)
            continue
        start_ts = float(row["window_start_ts"])
        end_ts = float(row["window_end_ts"])
        candidates = grp[(grp["survey_end_ts"] > start_ts) & (grp["survey_start_ts"] < end_ts)]
        if candidates.empty:
            labels.append(np.nan)
            overlaps.append(0.0)
            continue
        best_label = np.nan
        best_overlap = 0.0
        for _, interval in candidates.iterrows():
            interval_start = float(interval["survey_start_ts"]) + max(float(boundary_margin_sec), 0.0)
            interval_end = float(interval["survey_end_ts"]) - max(float(boundary_margin_sec), 0.0)
            if interval_end <= interval_start:
                interval_start = float(interval["survey_start_ts"])
                interval_end = float(interval["survey_end_ts"])
            overlap = max(0.0, min(end_ts, interval_end) - max(start_ts, interval_start))
            ratio = overlap / max(end_ts - start_ts, 1e-8)
            if ratio > best_overlap:
                best_overlap = ratio
                best_label = int(interval["nurse_label"])
        labels.append(best_label if best_overlap >= min_overlap else np.nan)
        overlaps.append(best_overlap)
    out["nurse_label"] = labels
    out["label_overlap_ratio"] = overlaps
    out["target3"] = out["nurse_label"]
    out["target_binary"] = out["nurse_label"].apply(lambda x: 1 if pd.notna(x) and int(x) >= 1 else (0 if pd.notna(x) else np.nan))
    return out


def build_nurse_features(config: ExtractionConfig) -> pd.DataFrame:
    if not config.nurse_zip.exists() or not config.survey_xlsx.exists():
        return pd.DataFrame()
    if config.survey_offset_hours is None:
        offset = choose_best_survey_offset(config.data_dir)
    else:
        offset = float(config.survey_offset_hours)
    intervals = load_survey_intervals(config.survey_xlsx, offset)
    frames = []
    count = 0
    with zipfile.ZipFile(config.nurse_zip) as zf:
        entries = _session_entries(config.nurse_zip)
        if config.max_nurse_sessions:
            entries = entries[: config.max_nurse_sessions]
        for info in entries:
            match = re.match(r"([^/]+)/([^/]+)\.zip$", info.filename)
            if not match:
                continue
            subject_id = normalize_subject_id(match.group(1))
            session_id = match.group(2)
            record = read_e4_zip_bytes(subject_id, session_id, zf.read(info.filename))
            if record is None:
                continue
            frame = extract_e4_features(record, config.window_sec, config.step_sec)
            if frame.empty:
                continue
            frames.append(frame)
            count += 1
            if config.max_nurse_sessions and count >= config.max_nurse_sessions:
                break
    if not frames:
        return pd.DataFrame()
    features = pd.concat(frames, ignore_index=True)
    features = _label_nurse_windows(
        features,
        intervals,
        config.min_label_overlap,
        config.label_boundary_margin_sec,
    )
    features["survey_offset_hours"] = offset
    if not config.keep_unlabeled:
        features = features[features["target3"].notna()].copy()
    return features


def build_combined_features(config: ExtractionConfig) -> pd.DataFrame:
    frames = []
    sources = set(config.sources)
    if "wesad" in sources:
        frames.append(build_wesad_features(config))
    if "nurse" in sources:
        frames.append(build_nurse_features(config))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = add_personal_baseline_features(combined, baseline_minutes=config.baseline_minutes)
    return combined
