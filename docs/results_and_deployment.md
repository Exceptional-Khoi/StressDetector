# Current Results And Deployment

This file reports the current scientific rerun saved under:

- `D:\IntroAI\StressDetector\outputs_scientific`

The active benchmark model set is exactly six algorithms for both binary and
3-class classification:

- Random Forest (`rf`)
- ExtraTrees (`extratrees`)
- XGBoost (`xgb`)
- LightGBM (`lgbm`)
- k-NN (`knn`)
- Gaussian Naive Bayes (`gnb`)

No additional algorithms are registered in the current benchmark.

## Extracted Dataset

Feature extraction was run on both datasets with the stricter scientific
preprocessing:

- WESAD: 526 labeled windows.
- Nurse wearable dataset: 4,019 labeled windows after stricter label cleaning.
- Total: 4,545 labeled windows.
- Feature table columns: 2,281.
- Model input features after excluding metadata/labels/timing columns: 2,257.

Unified `stress3` label distribution:

- `0`: 1,260 windows.
- `1`: 221 windows.
- `2`: 3,064 windows.

The nurse survey offset auto-selection selected `-4` hours as the best match
between sensor session timestamps and survey intervals.

## Current Combined Binary Benchmark

Protocol: subject-grouped folds, WESAD+nurse combined, nurse-only
calibration/threshold tuning, per-subject per-class cap 150, SMOTE inside each
training fold only, sigmoid holdout calibration, threshold grid over balanced
accuracy, macro F1, and recall-floor balanced accuracy.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.8102 | 0.7309 | 0.7328 | 0.8096 |
| XGBoost | 0.7253 | 0.7103 | 0.6564 | 0.7309 |
| LightGBM | 0.7288 | 0.7083 | 0.6502 | 0.7249 |
| ExtraTrees | 0.7100 | 0.6998 | 0.6321 | 0.7124 |
| k-NN | 0.6677 | 0.6985 | 0.6122 | 0.6957 |
| Gaussian Naive Bayes | 0.4461 | 0.5615 | 0.4262 | 0.4641 |

Deploy bundle:

- `D:\IntroAI\StressDetector\outputs_scientific\models\binary_best_model.joblib`
- Selected model: Random Forest.
- Selection metric: benchmark weighted F1.
- Final deploy threshold: `0.79`.

## Current Combined 3-Class Benchmark

Protocol: subject-grouped folds, WESAD+nurse combined, same feature set and
calibration policy. Binary threshold tuning is not used for 3-class prediction.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| LightGBM | 0.7667 | 0.6234 | 0.6079 | 0.7531 |
| Random Forest | 0.7688 | 0.6141 | 0.5889 | 0.7506 |
| ExtraTrees | 0.7543 | 0.6109 | 0.5773 | 0.7400 |
| XGBoost | 0.7510 | 0.6067 | 0.5850 | 0.7394 |
| k-NN | 0.6664 | 0.5880 | 0.4420 | 0.6799 |
| Gaussian Naive Bayes | 0.6508 | 0.4000 | 0.3062 | 0.5267 |

Deploy bundle:

- `D:\IntroAI\StressDetector\outputs_scientific\models\stress3_best_model.joblib`
- Selected model: LightGBM.
- Selection metric: benchmark weighted F1.
- Final deploy threshold: not applicable for 3-class.

## Current Nurse-Only Binary Benchmark

This is the source-specific deployment/reference benchmark for the nurse dataset
only. It uses the same six-model set and binary imbalance handling.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| LightGBM | 0.7391 | 0.4997 | 0.4927 | 0.7187 |
| XGBoost | 0.7145 | 0.5014 | 0.5075 | 0.7141 |
| ExtraTrees | 0.7217 | 0.5152 | 0.5039 | 0.7074 |
| Random Forest | 0.6674 | 0.4821 | 0.4806 | 0.6737 |
| k-NN | 0.6049 | 0.6252 | 0.5393 | 0.6557 |
| Gaussian Naive Bayes | 0.5940 | 0.6128 | 0.5102 | 0.6425 |

Deploy bundle:

- `D:\IntroAI\StressDetector\outputs_scientific\nurse_binary\models\binary_best_model.joblib`
- Selected model: LightGBM by benchmark weighted F1.
- Final deploy threshold: `0.45`.

If balanced accuracy is preferred over weighted F1 for nurse-only monitoring,
k-NN is the strongest current benchmark model, but the saved `best_model` follows
the project-wide weighted-F1 selection rule.

## Scientific Preprocessing And Improvements

The current rerun includes:

- event-based IBI/HRV features computed directly from `IBI.csv` events,
- direct E4 HR plus BVP-derived heart features,
- BVP/IBI/HR signal-quality and consistency features,
- 60s physiological windows plus trailing 120s and 300s summaries,
- robust personalized baseline deltas/ratios/z-scores with clipping,
- ACC-based physical-activity context features,
- stricter nurse label cleaning with `min_label_overlap = 0.70` and 60s survey
  boundary margin,
- subject-grouped evaluation to reduce subject leakage,
- temporal smoothing and rule-based decision support after classifier prediction.

## Deploy Bundle Contents

Each `.joblib` bundle stores:

- fitted model,
- median imputer,
- optional scaler,
- label encoder,
- exact feature column list,
- selected feature indices/columns,
- label semantics,
- baseline policy metadata,
- signal unit metadata,
- class-cap/resampling/calibration settings,
- binary decision threshold when applicable,
- decision-support metadata.

The current bundles were trained with `scikit-learn==1.7.2`; use the same
version in the backend environment when loading them.

Backend loading example:

```python
import pandas as pd
from stress_benchmark.deploy import load_bundle, predict_feature_frame

bundle = load_bundle(r"D:\IntroAI\StressDetector\outputs_scientific\models\binary_best_model.joblib")
features = pd.read_csv(r"D:\IntroAI\StressDetector\outputs_scientific\features_combined.csv.gz").head(10)
predictions = predict_feature_frame(bundle, features)
```
