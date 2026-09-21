#!/usr/bin/env python3
"""Evaluate all trained climate prediction models on the test set.

Loads models saved in models/<model_name> and calculates:
- Standard metrics: MAE, RMSE, R²
- IMD Rainfall category accuracy:
  * Dry (< 2.5 mm)
  * Moderate (2.5 - 15.5 mm)
  * Heavy (> 15.5 mm)

Usage:
    python -m climate_twin.training.evaluate_models
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from climate_twin.models import load_model, list_available_models
from climate_twin.training.train_all_models import load_and_split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_rainfall_categories(
    actual_rainfall: np.ndarray,
    predicted_rainfall: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate performance across IMD rainfall intensity categories."""
    categories = {
        "Dry (<2.5mm)": actual_rainfall < 2.5,
        "Moderate (2.5-15.5mm)": (actual_rainfall >= 2.5) & (actual_rainfall <= 15.5),
        "Heavy (>15.5mm)": actual_rainfall > 15.5,
    }

    results = {}
    for cat_name, mask in categories.items():
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue
        mae = float(mean_absolute_error(actual_rainfall[mask], predicted_rainfall[mask]))
        rmse = float(np.sqrt(mean_squared_error(actual_rainfall[mask], predicted_rainfall[mask])))
        results[cat_name] = {
            "sample_count": n_samples,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }
    return results


def main() -> int:
    logger.info("Loading test dataset...")
    data = load_and_split_data()
    X_test = data["X_test"]
    y_test = data["y_test"]

    available_models = list_available_models()
    evaluated_rows = []
    category_breakdowns = {}

    print("\n" + "=" * 80)
    print("DETAILED MULTI-MODEL TEST EVALUATION (IMD DATA)")
    print("=" * 80)

    for model_name in available_models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            logger.warning(f"Model directory not found for {model_name}: {model_path}")
            continue

        try:
            model = load_model(model_name, model_path)
            if model_name == "lstm":
                lookback = model._hyperparameters.get("lookback", 14)
                X_comb = np.vstack([data["X_val"], data["X_test"]])
                preds = model.predict(X_comb)
                test_start = max(0, len(preds) - len(y_test))
                preds = preds[test_start:]
                min_len = min(len(preds), len(y_test))
                preds = preds[:min_len]
                y_eval = y_test[:min_len]
            else:
                preds = model.predict(X_test)
                y_eval = y_test

            for i, target in enumerate(TARGET_COLUMNS):
                mae = mean_absolute_error(y_eval[:, i], preds[:, i])
                rmse = np.sqrt(mean_squared_error(y_eval[:, i], preds[:, i]))
                r2 = r2_score(y_eval[:, i], preds[:, i])
                evaluated_rows.append({
                    "Model": model.display_name,
                    "Target": target,
                    "MAE": round(float(mae), 4),
                    "RMSE": round(float(rmse), 4),
                    "R²": round(float(r2), 4),
                    "Training Time (s)": round(model.training_time_s, 2),
                })

            cat_eval = evaluate_rainfall_categories(y_eval[:, 0], preds[:, 0])
            category_breakdowns[model.display_name] = cat_eval

        except Exception as exc:
            logger.error(f"Error evaluating {model_name}: {exc}")

    df = pd.DataFrame(evaluated_rows)
    print("\n" + df.to_string(index=False))

    print("\n" + "=" * 80)
    print("IMD RAINFALL CATEGORY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    for model_name, cat_dict in category_breakdowns.items():
        print(f"\n--- {model_name} ---")
        for cat, scores in cat_dict.items():
            print(f"  {cat} (N={scores['sample_count']}): MAE = {scores['mae']} mm, RMSE = {scores['rmse']} mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
