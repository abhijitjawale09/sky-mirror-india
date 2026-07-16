# Sky Mirror India

An AI-powered digital twin of India's climate built with Flask, modern web visualization, and an extensible prediction stack for rainfall and temperature analysis.

## What this scaffold includes

* A Flask app factory with `/`, `/api/predict`, and `/health` endpoints.
* A climate twin engine that generates pilot-region observations and trains a gradient boosting forecaster.
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

This scaffold is ready to ingest IMD and MOSDAC datasets for rainfall, max temperature, min temperature, INSAT LST, and SST. The current implementation uses deterministic demo data so the dashboard works immediately, and the service layer is structured to swap in real national datasets without changing the UI contract.

## Model approach

The default forecasting layer uses temporal feature engineering with `HistGradientBoostingRegressor` for a strong tabular baseline. The code is organized so a deep temporal model can be added later for sequence learning and ensemble forecasting.
