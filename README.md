# RAN KPI AI (ranai)

LTE-first pipeline for RAN KPI load forecasting and anomaly detection, with
WCDMA/GSM support planned after the LTE data foundation is validated. The
project follows the product requirements: reproducible data preparation,
seasonal-naive benchmarking, time- and cell-held-out validation, and actionable
evaluation outputs. It does not automate network parameter changes.

## Setup

1. Create and activate a virtual environment.
2. Install the project in editable mode: `python -m pip install -e ".[dev]"`.
3. Configure local paths and optional database settings in `.env`.
4. Put source exports under `data/raw/kpi/lte/`, configuration under
   `data/raw/config/`, and change records under `data/raw/change_log/`.
   Raw data, derived data, models, and MLflow runs are excluded from Git.
5. Map vendor-specific columns in `configs/data.yaml`; review feature and
   model settings in `configs/features.yaml` and `configs/model_lgbm.yaml`.

Input data should be normalized to one row per cell and timestamp. The project
does not assume a particular vendor schema or define operational thresholds:
confirm the KPI mapping, outage indicators, cell identifiers, history, and
degradation thresholds with the data owner before producing operational
forecasts or alerts.

## Project layout

- `data/raw/`: original KPI exports, configuration, and change logs.
- `data/interim/`, `data/processed/`: derived datasets.
- `configs/`: data, feature, and LightGBM settings.
- `src/ranai/`: reusable package code.
- `scripts/`: command-line pipeline entry points.
- `notebooks/`: exploratory analysis and experiments.
- `models/`, `mlruns/`, `reports/`: generated artifacts.
- MLflow uses a local SQLite tracking database at `mlruns/mlflow.db` by default.

## Development

Run tests with `pytest`. Dates use ISO `YYYY-MM-DD` format. Select `hourly` or
`daily` grain to match the input, and keep each cell's timestamp series sorted.
Lag and rolling features are computed from prior observations only. Evaluation
must use chronological splits; holdout cells are kept separate from training.

The pipeline intentionally starts with LTE. Forecast quality is compared with
a same-hour-last-week baseline before more complex models are accepted. Alert
precision, false alarms, and lead time still require review against engineer-
confirmed incidents; these thresholds are not assumed by this starter project.

Run stages from the project root after installing the package:

```powershell
python scripts/run_clean.py
python scripts/run_join.py
python scripts/run_features.py
python scripts/run_train.py
python scripts/run_evaluate.py reports/lte_evaluation_predictions.csv `
  --target-column prb_utilization
python scripts/run_score.py
```

Set `technology: lte`, vendor column mappings, KPI target, feature lists, and
held-out cell IDs in the YAML configs before running. For a first data check,
start with cleaning and inspect `data/interim/lte_data_quality.csv`.

## Current scope and data-dependent work

Reusable CSV/Excel loading, KPI cleaning/quality summaries, time and neighbor
features, leakage-aware splits, seasonal-naive comparison, recursive forecasts,
and forecast/alert metrics are provided without guessing vendor schemas. A
production run additionally needs the actual column mappings, outage/change
labels, target KPI, and agreed alert/degradation thresholds. Resolve the
remaining data-owner questions before scheduling or operationalizing scoring.
