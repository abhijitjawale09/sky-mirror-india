# Sky Mirror India

An AI-powered digital twin of India's climate built with Flask, modern web visualization, and an extensible prediction stack for rainfall and temperature analysis.

## What this scaffold includes

* A Flask app factory with `/`, `/api/predict`, and `/health` endpoints.
* A climate twin engine that trains on fused IMD observations and serves scenario-based forecasts.
* Scenario simulation controls for rainfall and temperature what-if analysis.
* A custom dashboard with a modern glass UI, interactive map, and forecast charts.

## Project structure

* `app.py` - local entrypoint.
* `climate_twin/` - Flask app, data generation, forecasting, and digital twin service layer.
* `templates/` - HTML templates for the dashboard.
* `static/` - CSS and client-side interactivity.
* `tests/` - basic smoke tests for the app.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --debug
```

## Data plan

This scaffold now trains on IMD rainfall, maximum temperature, and minimum temperature GRD data only. The pipeline downloads or reuses local GRD files, converts them to CSV, and writes a fused table to `data/processed/climate_training_data.csv` for the app and model.

### Scalable storage

For scalable downstream processing, convert the fused CSV into partitioned Parquet using:

```bash
python scripts/build_parquet.py
```

If `pyarrow` is installed the script will write partitioned Parquet by `year` and `region` under `data/processed/parquet/` — otherwise it will write per-year CSVs as a fallback.

### Validation

Run the validation harness which computes deterministic and ensemble metrics (MAE, RMSE, CRPS):

```bash
python scripts/validate_model.py
```

This writes `models/validation_report.json` containing deterministic and ensemble-based performance summaries.

## Model approach

The forecasting layer now uses a Random Forest multi-output regressor trained on fused IMD rainfall, maximum temperature, and minimum temperature CSVs from your Downloads folder. The preprocessing pipeline builds a cached table in `data/processed/climate_training_data.csv`, so the app uses only the real IMD-derived observations and does not depend on synthetic demo data.

## Training script

Run the full preprocess and training flow with:

```bash
python scripts/train_model.py
```

This will regenerate the processed dataset and save a trained Random Forest model under `models/random_forest_model.pkl`.

## IMD 15-year pipeline

To download and convert the previous 15 years of IMD rainfall, maximum temperature, and minimum temperature GRD files into the processed CSV used by the app, run:

```bash
python scripts/download_imd.py --start 2011 --end 2025
```

The pipeline first reuses any matching GRD files already present in `~/Downloads` or the project cache, then downloads missing files from the IMD pages, converts them to CSV, and writes the fused training table to `data/processed/climate_training_data.csv`.
