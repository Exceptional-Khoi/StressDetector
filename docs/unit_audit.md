# Unit Audit

Sources checked:

- WESAD official page: `https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html`
- Nurse Dryad dataset page: `https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f`
- Local WESAD readme PDF inside `WESAD.zip`.
- Local Empatica E4 `info.txt` files inside nurse session zip files.

## Common E4 Modalities

The shared wrist-worn Empatica E4 signals are:

- ACC
- BVP
- EDA
- TEMP

Both datasets include raw Empatica E4 exports with E4-derived:

- HR
- IBI

WESAD `.pkl` wrist data exposes only ACC/BVP/EDA/TEMP, but the companion
`S*_E4_Data.zip` files contain `HR.csv` and `IBI.csv`. The pipeline now reads
those raw E4 files and aligns/crops HR/IBI to the synchronized WESAD `.pkl`
timeline before feature extraction.

## Unit Compatibility

| Signal | WESAD local check | Nurse local check | Pipeline handling |
|---|---|---|---|
| ACC | values in `[-128, 127]` | values in `[-128, 127]` | both converted from Empatica `1/64 g` to `g` |
| EDA | microsiemens-scale values | Dryad says electrodermal activity/electrical conductivity; local values match microsiemens scale | used as microsiemens |
| TEMP | Celsius-like skin temperature | Dryad explicitly says Celsius | used as Celsius |
| BVP | Empatica PPG amplitude | Dryad says blood volume pulse; local scale matches Empatica amplitude | used as raw amplitude plus BVP-derived HR/HRV |
| HR | available in raw `S*_E4_Data.zip/HR.csv`; average heart rate extracted from BVP at 1 Hz | same E4 `HR.csv` format; starts 10 seconds after other E4 files | aligned/cropped and used as a common HR feature |
| IBI | available in raw `S*_E4_Data.zip/IBI.csv`; seconds between beats extracted from BVP | same E4 `IBI.csv` format; seconds between beats extracted from BVP | aligned/cropped and used as a common IBI/HRV feature |

## Important Fixes

- ACC is normalized for both datasets. Earlier raw checks confirmed both WESAD `.pkl`
  wrist ACC and nurse E4 ACC use the same `[-128, 127]` Empatica scale.
- HR in both E4 exports starts 10 seconds after ACC/EDA/BVP/TEMP. The reader pads
  HR to align all signals to the same session start.
- WESAD raw E4 HR/IBI are now used. The WESAD loader estimates the `.pkl` crop
  offset by matching raw E4 BVP/ACC/EDA/TEMP sequences to the synchronized `.pkl`
  arrays, then crops HR/IBI to the same timeline.
- Model input excludes time/order columns and survey-offset columns.

## Advanced Features Added

Common features:

- BVP-derived HR.
- BVP-derived HRV: RR mean/std/median/range, SDNN, RMSSD, pNN50, CVNN, LF/HF approximations.
- Direct E4 HR statistics from `HR.csv`.
- Direct E4 IBI statistics and IBI-derived HRV from `IBI.csv`.
- EDA tonic/phasic approximation.
- EDA SCR peak count/rate/amplitude.
- ACC magnitude/dynamic magnitude/jerk.
- ACC stationary ratio and active ratio.
- Personalized baseline deltas/ratios/z-scores for all numeric features.

Nurse-specific additional features:

- None among the E4 physiological streams used here; HR/IBI are now common to
  both WESAD raw E4 and the nurse raw E4 sessions.
