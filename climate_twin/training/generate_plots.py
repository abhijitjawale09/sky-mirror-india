#!/usr/bin/env python3
"""Generate professional comparison plots from model evaluation results.

Creates all required visualizations for research presentation:
- Model performance bar charts
- Actual vs predicted scatter plots
- Residual analysis
- Error distributions
- Feature importance
- Training time comparison

Usage:
    python -m climate_twin.training.generate_plots
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "prediction_results"

# Professional color palette
MODEL_COLORS = {
    "random_forest": "#2196F3",
    "xgboost": "#FF9800",
    "hist_gradient_boosting": "#4CAF50",
    "lstm": "#9C27B0",
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting",
    "lstm": "LSTM",
}

TARGET_LABELS = {
    "rainfall_mm": "Rainfall (mm)",
    "tmax_c": "Max Temperature (°C)",
    "tmin_c": "Min Temperature (°C)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def load_comparison_data() -> pd.DataFrame:
    """Load model comparison CSV."""
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run train_all_models.py first: {path}")
    return pd.read_csv(path)


def load_predictions(model_name: str) -> pd.DataFrame | None:
    """Load saved predictions for a model."""
    path = PREDICTIONS_DIR / f"{model_name}_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_performance_bars(df: pd.DataFrame) -> None:
    """A. Model Performance Bar Charts — MAE, RMSE, R² per target."""
    for metric, metric_label in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R²", "R² Score")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        fig.suptitle(f"Model Comparison — {metric_label}", fontsize=16, fontweight="bold", y=1.02)

        for ax, target in zip(axes, ["rainfall_mm", "tmax_c", "tmin_c"]):
            subset = df[df["Target"] == target].sort_values(metric)
            models = subset["Model"].tolist()
            values = subset[metric].tolist()
            colors = [MODEL_COLORS.get(m, "#999") for m in models]
            labels = [MODEL_LABELS.get(m, m) for m in models]

            bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
            ax.set_title(TARGET_LABELS.get(target, target), fontweight="bold")
            ax.set_xlabel(metric_label)

            # Add value annotations
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"performance_bars_{metric.lower().replace('²', '2')}.png")
        plt.close()
        logger.info(f"Saved performance_bars_{metric.lower().replace('²', '2')}.png")


def plot_actual_vs_predicted(df: pd.DataFrame) -> None:
    """B. Actual vs Predicted scatter plots for each target × model."""
    models_with_preds = []
    for model_name in df["Model"].unique():
        preds = load_predictions(model_name)
        if preds is not None:
            models_with_preds.append((model_name, preds))

    if not models_with_preds:
        logger.warning("No prediction files found. Skipping actual vs predicted plots.")
        return

    for target_col, target_label in [
        ("rainfall_mm", "Rainfall (mm)"),
        ("tmax_c", "Max Temperature (°C)"),
        ("tmin_c", "Min Temperature (°C)"),
    ]:
        n_models = len(models_with_preds)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
        fig.suptitle(f"Actual vs Predicted — {target_label}", fontsize=16, fontweight="bold", y=1.02)

        for idx, (model_name, preds) in enumerate(models_with_preds):
            ax = axes[0, idx]
            actual = preds[f"actual_{target_col}"]
            predicted = preds[f"predicted_{target_col}"]

            ax.scatter(actual, predicted, alpha=0.3, s=10,
                      color=MODEL_COLORS.get(model_name, "#999"), edgecolors="none")

            # Perfect prediction line
            lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect")

            ax.set_xlabel(f"Actual {target_label}")
            ax.set_ylabel(f"Predicted {target_label}")
            ax.set_title(MODEL_LABELS.get(model_name, model_name), fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"actual_vs_predicted_{target_col}.png")
        plt.close()
        logger.info(f"Saved actual_vs_predicted_{target_col}.png")


def plot_residuals(df: pd.DataFrame) -> None:
    """D. Residual analysis — prediction errors."""
    models_with_preds = []
    for model_name in df["Model"].unique():
        preds = load_predictions(model_name)
        if preds is not None:
            models_with_preds.append((model_name, preds))

    if not models_with_preds:
        return

    for target_col, target_label in [
        ("rainfall_mm", "Rainfall (mm)"),
        ("tmax_c", "Max Temperature (°C)"),
        ("tmin_c", "Min Temperature (°C)"),
    ]:
        n_models = len(models_with_preds)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
        fig.suptitle(f"Residual Distribution — {target_label}", fontsize=16, fontweight="bold", y=1.02)

        for idx, (model_name, preds) in enumerate(models_with_preds):
            ax = axes[0, idx]
            residuals = preds[f"predicted_{target_col}"] - preds[f"actual_{target_col}"]

            ax.hist(residuals, bins=40, color=MODEL_COLORS.get(model_name, "#999"),
                   alpha=0.7, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax.axvline(residuals.mean(), color="red", linewidth=1, linestyle="-",
                      alpha=0.8, label=f"Mean: {residuals.mean():.3f}")

            ax.set_xlabel(f"Error ({target_label})")
            ax.set_ylabel("Count")
            ax.set_title(MODEL_LABELS.get(model_name, model_name), fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"residuals_{target_col}.png")
        plt.close()
        logger.info(f"Saved residuals_{target_col}.png")


def plot_error_distribution_comparison() -> None:
    """E. Overlay error distributions from all models on one plot."""
    comparison = load_comparison_data()
    models = comparison["Model"].unique()

    for target_col, target_label in [
        ("rainfall_mm", "Rainfall (mm)"),
        ("tmax_c", "Max Temperature (°C)"),
        ("tmin_c", "Min Temperature (°C)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Error Distribution Comparison — {target_label}",
                    fontsize=14, fontweight="bold")

        for model_name in models:
            preds = load_predictions(model_name)
            if preds is None:
                continue
            errors = np.abs(preds[f"predicted_{target_col}"] - preds[f"actual_{target_col}"])
            ax.hist(errors, bins=40, alpha=0.4,
                   color=MODEL_COLORS.get(model_name, "#999"),
                   label=MODEL_LABELS.get(model_name, model_name),
                   edgecolor="white", linewidth=0.3)

        ax.set_xlabel(f"Absolute Error ({target_label})")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"error_comparison_{target_col}.png")
        plt.close()
        logger.info(f"Saved error_comparison_{target_col}.png")


def plot_feature_importance() -> None:
    """F. Feature importance for tree-based models."""
    fi_path = RESULTS_DIR / "feature_importance.json"
    if not fi_path.exists():
        logger.warning("No feature importance data found.")
        return

    with open(fi_path) as f:
        fi_data = json.load(f)

    n_models = len(fi_data)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), squeeze=False)
    fig.suptitle("Feature Importance Comparison", fontsize=16, fontweight="bold", y=1.01)

    for idx, (model_name, importances) in enumerate(fi_data.items()):
        ax = axes[0, idx]
        features = list(importances.keys())
        values = list(importances.values())

        # Sort by importance
        sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
        features = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        bars = ax.barh(features, values,
                      color=MODEL_COLORS.get(model_name, "#999"),
                      edgecolor="white", linewidth=0.5, height=0.7)

        ax.set_xlabel("Importance (%)")
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()
    logger.info("Saved feature_importance.png")


def plot_training_time(df: pd.DataFrame) -> None:
    """G. Training time comparison."""
    # Get unique model training times
    time_data = df.drop_duplicates(subset=["Model"])[["Model", "Training Time (s)"]].copy()
    time_data = time_data.sort_values("Training Time (s)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")

    models = time_data["Model"].tolist()
    times = time_data["Training Time (s)"].tolist()
    colors = [MODEL_COLORS.get(m, "#999") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_time.png")
    plt.close()
    logger.info("Saved training_time.png")


def plot_time_series_predictions() -> None:
    """C. Time-series prediction overlay — actual vs all models over time."""
    comparison = load_comparison_data()
    models = comparison["Model"].unique()

    for target_col, target_label in [
        ("rainfall_mm", "Rainfall (mm)"),
        ("tmax_c", "Max Temperature (°C)"),
        ("tmin_c", "Min Temperature (°C)"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title(f"Time-Series Predictions — {target_label} (Test Period)",
                    fontsize=14, fontweight="bold")

        plotted = False
        for model_name in models:
            preds = load_predictions(model_name)
            if preds is None:
                continue

            # Plot actual (only once)
            if not plotted:
                ax.plot(preds[f"actual_{target_col}"].values, color="black",
                       linewidth=1.5, alpha=0.8, label="Actual", zorder=5)
                plotted = True

            ax.plot(preds[f"predicted_{target_col}"].values,
                   color=MODEL_COLORS.get(model_name, "#999"),
                   linewidth=1, alpha=0.7,
                   label=MODEL_LABELS.get(model_name, model_name))

        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel(target_label)
        ax.legend(loc="upper right", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"timeseries_{target_col}.png")
        plt.close()
        logger.info(f"Saved timeseries_{target_col}.png")


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_comparison_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info("Generating plots...")

    plot_performance_bars(df)
    plot_actual_vs_predicted(df)
    plot_time_series_predictions()
    plot_residuals(df)
    plot_error_distribution_comparison()
    plot_feature_importance()
    plot_training_time(df)

    logger.info(f"\nAll plots saved to: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
