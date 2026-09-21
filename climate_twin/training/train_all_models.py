#!/usr/bin/env python3
"""Train all climate prediction models using the same data split.

This script:
1. Loads the processed training data through the existing preprocessing pipeline.
2. Creates a chronological train/val/test split (70/15/15%).
3. Trains all 4 models (Random Forest, XGBoost, HistGradientBoosting, LSTM).
4. Saves trained models and per-target evaluation metrics.
5. Generates the comparison table and selects the best model per target.

Usage:
    python -m climate_twin.training.train_all_models
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.data.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    prepare_training_features,
)
from climate_twin.data.loader import load_training_observations
from climate_twin.models import (
    ClimateModel,
    ModelMetrics,
    create_model,
    list_available_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "prediction_results"


def load_and_split_data() -> dict[str, np.ndarray | pd.DataFrame]:
    """Load data through the existing pipeline and split chronologically."""
    logger.info("Loading training observations...")
    observations, data_source = load_training_observations()
    logger.info(f"Data source: {data_source}")
    logger.info(f"Observations shape: {observations.shape}")

    logger.info("Preparing features through existing preprocessing pipeline...")
    feature_frame, region_code_map = prepare_training_features(observations)
    logger.info(f"Feature frame shape: {feature_frame.shape}")
    logger.info(f"Feature columns: {FEATURE_COLUMNS}")
    logger.info(f"Target columns: {TARGET_COLUMNS}")

    # Chronological split: 70% train, 15% validation, 15% test
    n = len(feature_frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = feature_frame.iloc[:train_end]
    val = feature_frame.iloc[train_end:val_end]
    test = feature_frame.iloc[val_end:]

    logger.info(f"Split sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    X_train = train[FEATURE_COLUMNS].values
    y_train = train[TARGET_COLUMNS].values
    X_val = val[FEATURE_COLUMNS].values
    y_val = val[TARGET_COLUMNS].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test[TARGET_COLUMNS].values

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_frame": feature_frame,
        "region_code_map": region_code_map,
        "data_source": data_source,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }


def train_single_model(
    model_name: str,
    data: dict,
) -> tuple[ClimateModel, list[ModelMetrics]]:
    """Train a single model and evaluate it on the test set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'='*60}")

    model = create_model(model_name)

    # For LSTM: feed train+val sorted chronologically for sequence creation
    # For tree models: use standard train with val for early stopping
    try:
        model.timed_fit(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            feature_names=FEATURE_COLUMNS.copy(),
        )
    except Exception as exc:
        logger.error(f"Failed to train {model_name}: {exc}")
        raise

    logger.info(f"Training time: {model.training_time_s:.2f}s")

    # Evaluate on test set
    if model_name == "lstm":
        # LSTM needs sequential data for prediction — use the full test portion
        # We need to create sequences from val+test for LSTM evaluation
        X_combined = np.vstack([data["X_val"], data["X_test"]])
        y_combined = np.vstack([data["y_val"], data["y_test"]])
        preds = model.predict(X_combined)
        # Only evaluate on the test portion (after lookback sequences are created)
        lookback = model._hyperparameters.get("lookback", 14)
        test_start = max(0, len(preds) - data["y_test"].shape[0])
        y_test_eval = y_combined[lookback + test_start:]
        preds_eval = preds[test_start:]
        # Align lengths
        min_len = min(len(y_test_eval), len(preds_eval))
        if min_len > 0:
            y_test_eval = y_test_eval[:min_len]
            preds_eval = preds_eval[:min_len]
        else:
            logger.warning(f"LSTM: insufficient test sequences, using raw prediction")
            y_test_eval = data["y_test"]
            preds_eval = model.predict(np.vstack([data["X_train"][-lookback:], data["X_test"]]))[:]
            min_len = min(len(y_test_eval), len(preds_eval))
            y_test_eval = y_test_eval[:min_len]
            preds_eval = preds_eval[:min_len]

        metrics = []
        for i, target in enumerate(TARGET_COLUMNS):
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = float(mean_absolute_error(y_test_eval[:, i], preds_eval[:, i]))
            rmse = float(np.sqrt(mean_squared_error(y_test_eval[:, i], preds_eval[:, i])))
            r2 = float(r2_score(y_test_eval[:, i], preds_eval[:, i]))
            metrics.append(ModelMetrics(
                model_name=model_name,
                target=target,
                mae=mae, rmse=rmse, r2=r2,
                training_time_s=model.training_time_s,
                hyperparameters=model.get_hyperparameters(),
            ))
    else:
        metrics = model.evaluate(data["X_test"], data["y_test"])

    # Save model
    model_dir = MODELS_DIR / model_name
    model.save(model_dir)
    logger.info(f"Saved {model_name} to {model_dir}")

    for m in metrics:
        logger.info(f"  {m.target}: MAE={m.mae:.4f}, RMSE={m.rmse:.4f}, R²={m.r2:.4f}")

    return model, metrics


def select_best_models(
    all_metrics: list[ModelMetrics],
) -> dict[str, dict[str, Any]]:
    """Select the best model per target variable.

    Primary metric: MAE (lower is better).
    For rainfall, RMSE is also considered because of heavy-tail distribution.
    """
    best_models: dict[str, dict[str, Any]] = {}

    for target in TARGET_COLUMNS:
        target_metrics = [m for m in all_metrics if m.target == target]
        if not target_metrics:
            continue

        # Sort by MAE (primary metric)
        target_metrics.sort(key=lambda m: m.mae)
        best = target_metrics[0]

        best_models[target] = {
            "model_name": best.model_name,
            "mae": best.mae,
            "rmse": best.rmse,
            "r2": best.r2,
            "training_time_s": best.training_time_s,
            "reason": f"Lowest MAE ({best.mae:.4f}) among {len(target_metrics)} models",
        }

    return best_models


def generate_comparison_table(all_metrics: list[ModelMetrics]) -> pd.DataFrame:
    """Create a comparison DataFrame from all model metrics."""
    rows = []
    for m in all_metrics:
        rows.append({
            "Model": m.model_name,
            "Target": m.target,
            "MAE": round(m.mae, 4),
            "RMSE": round(m.rmse, 4),
            "R²": round(m.r2, 4),
            "Training Time (s)": round(m.training_time_s, 2),
        })
    return pd.DataFrame(rows)


def save_results(
    all_metrics: list[ModelMetrics],
    best_models: dict[str, dict],
    comparison_df: pd.DataFrame,
    data: dict,
    models: dict[str, ClimateModel],
) -> None:
    """Save all results to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics CSV
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved comparison table to {RESULTS_DIR / 'model_comparison.csv'}")

    # Save detailed metrics
    metrics_list = [m.to_dict() for m in all_metrics]
    with open(RESULTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Save best models
    with open(RESULTS_DIR / "best_models.json", "w") as f:
        json.dump(best_models, f, indent=2)
    logger.info(f"Saved best models to {RESULTS_DIR / 'best_models.json'}")

    # Save experiment config
    config = {
        "data_source": data["data_source"],
        "total_samples": data["train_size"] + data["val_size"] + data["test_size"],
        "train_size": data["train_size"],
        "val_size": data["val_size"],
        "test_size": data["test_size"],
        "split_method": "chronological_70_15_15",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "models_trained": list(models.keys()),
        "random_seed": 42,
    }
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save test predictions for each model
    for model_name, model in models.items():
        try:
            if model_name == "lstm":
                X_combined = np.vstack([data["X_val"], data["X_test"]])
                preds = model.predict(X_combined)
                lookback = model._hyperparameters.get("lookback", 14)
                test_start = max(0, len(preds) - data["y_test"].shape[0])
                preds = preds[test_start:]
                min_len = min(len(preds), data["y_test"].shape[0])
                preds = preds[:min_len]
                y_actual = data["y_test"][:min_len]
            else:
                preds = model.predict(data["X_test"])
                y_actual = data["y_test"]

            pred_df = pd.DataFrame({
                "actual_rainfall_mm": y_actual[:, 0],
                "predicted_rainfall_mm": preds[:, 0],
                "actual_tmax_c": y_actual[:, 1],
                "predicted_tmax_c": preds[:, 1],
                "actual_tmin_c": y_actual[:, 2],
                "predicted_tmin_c": preds[:, 2],
            })
            pred_df.to_csv(PREDICTIONS_DIR / f"{model_name}_predictions.csv", index=False)
        except Exception as exc:
            logger.warning(f"Could not save predictions for {model_name}: {exc}")

    # Save feature importance for models that support it
    importance_data = {}
    for model_name, model in models.items():
        fi = model.get_feature_importance()
        if fi is not None:
            importance_data[model_name] = fi
    if importance_data:
        with open(RESULTS_DIR / "feature_importance.json", "w") as f:
            json.dump(importance_data, f, indent=2)


def main() -> int:
    """Main training pipeline."""
    start_time = time.time()

    # Load and split data
    data = load_and_split_data()

    # Train all available models
    available_models = list_available_models()
    logger.info(f"Available models: {available_models}")

    trained_models: dict[str, ClimateModel] = {}
    all_metrics: list[ModelMetrics] = []

    for model_name in available_models:
        try:
            model, metrics = train_single_model(model_name, data)
            trained_models[model_name] = model
            all_metrics.extend(metrics)
        except Exception as exc:
            logger.error(f"Skipping {model_name} due to error: {exc}")
            continue

    if not all_metrics:
        logger.error("No models were successfully trained!")
        return 1

    # Generate comparison table
    comparison_df = generate_comparison_table(all_metrics)
    logger.info(f"\n{'='*80}")
    logger.info("MODEL COMPARISON TABLE")
    logger.info(f"{'='*80}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Select best models
    best_models = select_best_models(all_metrics)
    logger.info(f"\n{'='*80}")
    logger.info("BEST MODELS PER TARGET")
    logger.info(f"{'='*80}")
    for target, info in best_models.items():
        logger.info(f"  {target}: {info['model_name']} (MAE={info['mae']:.4f})")

    # Save everything
    save_results(all_metrics, best_models, comparison_df, data, trained_models)

    total_time = time.time() - start_time
    logger.info(f"\nTotal pipeline time: {total_time:.1f}s")
    logger.info(f"Results saved to: {RESULTS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
