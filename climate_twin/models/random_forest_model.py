"""Random Forest Regressor wrapper for the multi-model framework.

Wraps the existing Random Forest implementation into the standard ClimateModel
interface. Preserves the original 300-tree configuration as the baseline.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import ClimateModel


class RandomForestClimateModel(ClimateModel):
    """Random Forest multi-output regressor for climate prediction."""

    name = "random_forest"
    display_name = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 18,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.model: RandomForestRegressor | None = None
        self._hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
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
        self.model = RandomForestRegressor(
            n_estimators=self._hyperparameters["n_estimators"],
            max_depth=self._hyperparameters["max_depth"],
            min_samples_leaf=self._hyperparameters["min_samples_leaf"],
            n_jobs=-1,
            random_state=self._hyperparameters["random_state"],
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Random Forest model has not been trained.")
        return self.model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use individual tree predictions for 80% prediction intervals."""
        if self.model is None:
            raise RuntimeError("Random Forest model has not been trained.")

        tree_preds = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])  # (n_trees, n_samples, 3)

        mean_pred = np.mean(tree_preds, axis=0)
        lower = np.percentile(tree_preds, 10, axis=0)
        upper = np.percentile(tree_preds, 90, axis=0)
        return mean_pred, lower, upper

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_columns:
            return None
        importances = self.model.feature_importances_
        return {
            col: round(float(imp) * 100, 2)
            for col, imp in sorted(
                zip(self.feature_columns, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "random_forest_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "hyperparameters": self._hyperparameters,
                "training_time_s": self.training_time_s,
            }, f)
        return path

    def load(self, directory: Path) -> None:
        path = directory / "random_forest_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.training_time_s = data.get("training_time_s", 0.0)
        self.is_fitted = True
