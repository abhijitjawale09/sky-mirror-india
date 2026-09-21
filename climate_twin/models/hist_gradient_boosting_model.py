"""Histogram-based Gradient Boosting Regressor for climate prediction.

Uses scikit-learn's HistGradientBoostingRegressor which natively handles
NaN values (useful for INSAT columns), trains fast via histogram binning,
and provides a strong gradient-boosting comparison to XGBoost.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base_model import ClimateModel, TARGET_COLUMNS


class HistGBClimateModel(ClimateModel):
    """Histogram-based Gradient Boosting for climate prediction."""

    name = "hist_gradient_boosting"
    display_name = "HistGradientBoosting"

    def __init__(
        self,
        max_iter: int = 500,
        max_depth: int = 10,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 10,
        l2_regularization: float = 0.1,
        max_bins: int = 255,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self._hyperparameters = {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "max_bins": max_bins,
            "random_state": random_state,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_columns = feature_names or []

        for i, target in enumerate(TARGET_COLUMNS):
            # Use early stopping if validation data is available
            early_stopping = X_val is not None and y_val is not None

            model = HistGradientBoostingRegressor(
                max_iter=self._hyperparameters["max_iter"],
                max_depth=self._hyperparameters["max_depth"],
                learning_rate=self._hyperparameters["learning_rate"],
                min_samples_leaf=self._hyperparameters["min_samples_leaf"],
                l2_regularization=self._hyperparameters["l2_regularization"],
                max_bins=self._hyperparameters["max_bins"],
                random_state=self._hyperparameters["random_state"],
                early_stopping=early_stopping,
                validation_fraction=0.15 if not early_stopping else None,
                n_iter_no_change=20 if early_stopping else None,
            )

            model.fit(X_train, y_train[:, i])
            self.models[target] = model

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("HistGradientBoosting model has not been trained.")

        predictions = np.column_stack([
            self.models[target].predict(X) for target in TARGET_COLUMNS
        ])
        return predictions

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return permutation-style importance proxy from internal splits."""
        if not self.models or not self.feature_columns:
            return None

        # Use the number of times each feature was used in splits as proxy
        # HistGradientBoosting doesn't have .feature_importances_ like RF,
        # but sklearn >= 1.0 does expose it via the same attribute.
        try:
            all_importances = np.zeros(len(self.feature_columns))
            for target, model in self.models.items():
                # This is supported from sklearn 1.0+
                # Uses the mean decrease in impurity across all trees
                importances = np.zeros(len(self.feature_columns))
                # Access internal feature importances if available
                if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                    raw = model.feature_importances_
                    if len(raw) == len(self.feature_columns):
                        importances = raw
                all_importances += importances

            total = all_importances.sum()
            if total == 0:
                return None

            all_importances = all_importances / total * 100.0

            return {
                col: round(float(imp), 2)
                for col, imp in sorted(
                    zip(self.feature_columns, all_importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }
        except Exception:
            return None

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "hist_gb_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "feature_columns": self.feature_columns,
                "hyperparameters": self._hyperparameters,
                "training_time_s": self.training_time_s,
            }, f)
        return path

    def load(self, directory: Path) -> None:
        path = directory / "hist_gb_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.training_time_s = data.get("training_time_s", 0.0)
        self.is_fitted = True
