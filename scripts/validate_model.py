"""Validate the trained forecaster and compute ensemble-based uncertainty metrics.

Produces `models/validation_report.json` with MAE, RMSE and CRPS per variable.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.services.forecasting import ClimateForecaster


def crps_ensemble(obs: np.ndarray, ensemble: np.ndarray) -> float:
    """Compute CRPS for one observation against ensemble members.

    Uses the identity CRPS = E|X - x| - 0.5 E|X - X'| where X,X' are iid draws
    from the predictive distribution approximated by the ensemble.
    """
    # obs: scalar
    # ensemble: (n_members,)
    ensemble = np.asarray(ensemble)
    term1 = np.mean(np.abs(ensemble - obs))
    # pairwise absolute differences
    diffs = np.abs(ensemble.reshape(-1, 1) - ensemble.reshape(1, -1))
    term2 = 0.5 * np.mean(diffs)
    return float(term1 - term2)


def evaluate_with_ensemble(X_train, y_train, X_test, y_test, n_members=10):
    n_out = y_test.shape[1]
    preds_ensemble = np.zeros((n_members, X_test.shape[0], n_out))

    for m in range(n_members):
        model = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=100 + m)
        model.fit(X_train, y_train)
        preds_ensemble[m] = model.predict(X_test)

    # point predictions via ensemble mean
    mean_preds = preds_ensemble.mean(axis=0)

    metrics = {}
    names = ["rainfall_mm", "tmax_c", "tmin_c"]

    for i, name in enumerate(names):
        mae = float(mean_absolute_error(y_test[:, i], mean_preds[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_test[:, i], mean_preds[:, i])))

        # CRPS per sample, averaged
        crps_vals = [crps_ensemble(y_test[j, i], preds_ensemble[:, j, i]) for j in range(y_test.shape[0])]
        crps = float(np.mean(crps_vals)) if len(crps_vals) else None

        metrics[name] = {"mae": mae, "rmse": rmse, "crps": crps}

    return metrics


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "climate_training_data.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing training table: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"])

    forecaster = ClimateForecaster()
    prepared = forecaster.prepare_features(df)

    # time-based split: first 85% train, last 15% test
    split_idx = max(30, int(len(prepared) * 0.85))
    train = prepared.iloc[:split_idx]
    test = prepared.iloc[split_idx:]

    X_train = train[forecaster.feature_columns].values
    y_train = train[["rainfall_mm", "tmax_c", "tmin_c"]].values
    X_test = test[forecaster.feature_columns].values
    y_test = test[["rainfall_mm", "tmax_c", "tmin_c"]].values

    # single deterministic baseline (retrain same as production)
    baseline = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=42)
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)

    report = {
        "deterministic": {
            "rainfall_mm": {
                "mae": float(mean_absolute_error(y_test[:, 0], preds[:, 0])),
                "rmse": float(np.sqrt(mean_squared_error(y_test[:, 0], preds[:, 0]))),
            },
            "tmax_c": {
                "mae": float(mean_absolute_error(y_test[:, 1], preds[:, 1])),
                "rmse": float(np.sqrt(mean_squared_error(y_test[:, 1], preds[:, 1]))),
            },
            "tmin_c": {
                "mae": float(mean_absolute_error(y_test[:, 2], preds[:, 2])),
                "rmse": float(np.sqrt(mean_squared_error(y_test[:, 2], preds[:, 2]))),
            },
        }
    }

    # ensemble evaluation for uncertainty
    ensemble_metrics = evaluate_with_ensemble(X_train, y_train, X_test, y_test, n_members=12)
    report["ensemble"] = ensemble_metrics

    out_path = root / "models" / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote validation report to {out_path}")


if __name__ == "__main__":
    main()
