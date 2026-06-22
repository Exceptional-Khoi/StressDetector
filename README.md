# Stress ML Benchmark Pipeline

This project builds a reproducible ML benchmark for two wearable stress datasets:

- `WESAD.zip`: WESAD laboratory stress/affect dataset.
- `Stress_dataset.zip` + `SurveyResults.xlsx`: nurse wearable stress dataset with survey labels.

The code treats them as one benchmark family while preserving dataset provenance,
subject-level splits, and source-specific label rules.

## What Is Implemented

- Streaming readers for zipped Empatica E4 sessions.
- WESAD subject-by-subject feature extraction from synchronized `.pkl` files.
- Nurse dataset label joining from `SurveyResults.xlsx`.
- 60-second physiological feature windows inspired by WESAD.
- One-minute aggregation and class-imbalance handling inspired by the nurse stress paper.
- Personal baseline normalization per subject.
- ACC-based physical-activity context features.
- Leakage-safe subject-level benchmark splits.
- Classical ML benchmark: Random Forest, Extra Trees, XGBoost, LightGBM, k-NN, and Gaussian Naive Bayes.
- Binary imbalance handling: per-subject per-class cap, SMOTE, sigmoid calibration, and threshold tuning.
- Optional ablations: top-k feature selection, class-cap grids, threshold-metric grids, and source-balanced training.
- Decision-support post-processing with alert levels and simple recommendations.

## Setup

Install dependencies in a Python environment:

```powershell
pip install -r requirements.txt
```

The full benchmark expects `scikit-learn`, `imbalanced-learn`, `xgboost`, and
`lightgbm` in addition to the data-processing packages listed in `requirements.txt`.
If a requested model is unavailable, the pipeline stops instead of silently
running a reduced benchmark.
For deployment, keep the Python package versions consistent with the training
environment. The latest `outputs_scientific` bundles were trained with
`scikit-learn==1.7.2`;
loading them with a different scikit-learn version can produce compatibility
warnings or different behavior.

## Quick Run

Scan likely timezone offsets for the nurse survey labels:

```powershell
python -m stress_benchmark.cli scan-offsets --data-dir D:\IntroAI
```

Extract combined features:

```powershell
python -m stress_benchmark.cli extract --data-dir D:\IntroAI --out-dir D:\IntroAI\StressDetector\outputs --survey-offset auto
```

Run the benchmark:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\StressDetector\outputs --survey-offset auto --task binary --protocol groupkfold --models rf,extratrees,xgb,lgbm,knn,gnb
```

For binary tasks, class cap `150` and threshold tuning by balanced accuracy are enabled
by default. Use `--no-class-cap` or `--no-threshold-tuning` only for ablation.

Run the optional tuning/ablation grid:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\StressDetector\outputs --survey-offset auto --task binary --protocol groupkfold --models rf,extratrees,xgb,lgbm,knn,gnb --class-cap-grid 100,150,250 --feature-k-grid 200,300,all --threshold-metric-grid balanced_accuracy,macro_f1,balanced_accuracy_recall_floor --source-balance source_class
```

For the stricter scientific setting, use leave-one-subject-out:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\StressDetector\outputs --survey-offset auto --task stress3 --protocol loso
```

Train deployable backend bundles after feature extraction/benchmarking:

```powershell
python -m stress_benchmark.cli train-final --data-dir D:\IntroAI --out-dir D:\IntroAI\StressDetector\outputs --task binary --models rf,extratrees,xgb,lgbm,knn,gnb
```

Backend loading example:

```python
import pandas as pd
from stress_benchmark.deploy import load_bundle, predict_feature_frame

bundle = load_bundle(r"D:\IntroAI\StressDetector\outputs\models\binary_best_model.joblib")
features = pd.read_csv(r"D:\IntroAI\StressDetector\outputs\features_combined.csv.gz").head(10)
predictions = predict_feature_frame(bundle, features)
```

Run the local demo API:

```powershell
.\run_demo.bat
```

Then open `http://127.0.0.1:8000/`. By default the API loads
`outputs_scientific/models/binary_best_model.joblib`. To deploy another bundle,
set `STRESS_MODEL_PATH` before starting the server.

## Outputs

The benchmark writes:

- `outputs/features_combined.csv.gz`: extracted feature table.
- `outputs/metrics_<task>.csv`: fold/model metrics.
- `outputs/predictions_<task>.csv.gz`: row-level predictions and alert states.
- `outputs/summary_<task>.json`: aggregate metrics and settings.
- `outputs/models/<task>_<model>.joblib`: deployable model bundles.
- `outputs/models/<task>_best_model.joblib`: copy of the best bundle selected from benchmark weighted F1 when available.
- `outputs/models/manifest_<task>.json`: deployment manifest.

Current deploy bundles:

- Latest scientific combined binary:
  `D:\IntroAI\StressDetector\outputs_scientific\models\binary_best_model.joblib`
  uses Random Forest with threshold `0.79`.
- Latest scientific combined 3-class:
  `D:\IntroAI\StressDetector\outputs_scientific\models\stress3_best_model.joblib`
  uses LightGBM.
- Latest scientific nurse-only binary:
  `D:\IntroAI\StressDetector\outputs_scientific\nurse_binary\models\binary_best_model.joblib`
  uses LightGBM with threshold `0.45`.

- Combined WESAD+nurse binary: `D:\IntroAI\StressDetector\outputs\models\binary_best_model.joblib`
  uses ExtraTrees with threshold `0.70`.
- Nurse-only binary: `D:\IntroAI\StressDetector\outputs\nurse_binary\models\binary_best_model.joblib`
  uses ExtraTrees with threshold `0.83`.

Earlier WESAD HR/IBI rerun:

- `D:\IntroAI\StressDetector\outputs_hribi\models\binary_best_model.joblib`
  uses LightGBM with threshold `0.88`, selected by benchmark weighted F1.
  This bundle uses WESAD `HR.csv`/`IBI.csv` from `S*_E4_Data.zip` in addition
  to the original wrist `.pkl` ACC/BVP/EDA/TEMP streams.
- For highest balanced accuracy in this rerun, use
  `D:\IntroAI\StressDetector\outputs_hribi\models\binary_extratrees.joblib`
  with threshold `0.78`.

## Methodology

See `docs/methodology.md` for the paper-derived design choices and the added
personalized decision-support layer.
