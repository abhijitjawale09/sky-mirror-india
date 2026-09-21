"""XGBoost Regressor for climate prediction.

Uses XGBoost with sensible regularization to prevent overfitting on the
relatively small 2-year IMD dataset. Trains one XGBRegressor per target
variable because XGBoost does not natively support multi-output regression.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from .base_model import ClimateModel, TARGET_COLUMNS


class XGBoostClimateModel(ClimateModel):
    """XGBoost gradient boosting regressor for climate prediction."""

    name = "xgboost"
    display_name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.models: dict[str, XGBRegressor] = {}
        self._hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
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
            model = XGBRegressor(
                n_estimators=self._hyperparameters["n_estimators"],
                max_depth=self._hyperparameters["max_depth"],
                learning_rate=self._hyperparameters["learning_rate"],
                subsample=self._hyperparameters["subsample"],
                colsample_bytree=self._hyperparameters["colsample_bytree"],
                reg_alpha=self._hyperparameters["reg_alpha"],
                reg_lambda=self._hyperparameters["reg_lambda"],
                min_child_weight=self._hyperparameters["min_child_weight"],
                random_state=self._hyperparameters["random_state"],
                n_jobs=-1,
                verbosity=0,
            )

            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[:, i])]

            model.fit(
                X_train,
                y_train[:, i],
                eval_set=eval_set,
                verbose=False,
            )
            self.models[target] = model

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("XGBoost model has not been trained.")

        predictions = np.column_stack([
            self.models[target].predict(X) for target in TARGET_COLUMNS
        ])
        return predictions

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.models or not self.feature_columns:
            return None

        # Average feature importance across all 3 target models
        all_importances = np.zeros(len(self.feature_columns))
        for target, model in self.models.items():
            importances = model.feature_importances_
            all_importances += importances

        all_importances /= len(self.models)
        all_importances = all_importances / all_importances.sum() * 100.0

        return {
            col: round(float(imp), 2)
            for col, imp in sorted(
                zip(self.feature_columns, all_importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "xgboost_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "feature_columns": self.feature_columns,
                "hyperparameters": self._hyperparameters,
                "training_time_s": self.training_time_s,
            }, f)
        return path

    def load(self, directory: Path) -> None:
        path = directory / "xgboost_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.training_time_s = data.get("training_time_s", 0.0)
        self.is_fitted = True
