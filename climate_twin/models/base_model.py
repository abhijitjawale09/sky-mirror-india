"""Abstract base class for all climate prediction models.

Every model in the multi-model comparison framework implements this interface
so that training, evaluation, and dashboard integration remain consistent.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COLUMNS = ["rainfall_mm", "tmax_c", "tmin_c"]


@dataclass
class ModelMetrics:
    """Container for evaluation metrics of a single model on a single target."""

    model_name: str
    target: str
    mae: float
    rmse: float
    r2: float
    training_time_s: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target": self.target,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "training_time_s": round(self.training_time_s, 2),
            "hyperparameters": self.hyperparameters,
        }


class ClimateModel(ABC):
    """Abstract base class for climate prediction models.

    All models must implement fit, predict, save, load, and evaluate.
    Feature importance is optional (only meaningful for tree-based models).
    """

    name: str = "base"
    display_name: str = "Base Model"

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.training_time_s: float = 0.0
        self.feature_columns: list[str] = []
        self._hyperparameters: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the model. X and y are numpy arrays."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a numpy array of shape (n_samples, 3)."""
        ...

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Save model artifacts to directory. Return path to primary file."""
        ...

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load a previously saved model from directory."""
        ...

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances as {name: percentage}. None if unsupported."""
        return None

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the hyperparameters used for this model."""
        return self._hyperparameters.copy()

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: list[str] | None = None,
    ) -> list[ModelMetrics]:
        """Evaluate the model on test data and return per-target metrics."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.display_name} has not been trained.")

        if target_names is None:
            target_names = TARGET_COLUMNS

        predictions = self.predict(X_test)
        results: list[ModelMetrics] = []

        for i, target in enumerate(target_names):
            y_true = y_test[:, i]
            y_pred = predictions[:, i]

            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))

            results.append(
                ModelMetrics(
                    model_name=self.name,
                    target=target,
                    mae=mae,
                    rmse=rmse,
                    r2=r2,
                    training_time_s=self.training_time_s,
                    hyperparameters=self.get_hyperparameters(),
                )
            )

        return results

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower_80, upper_80). Default: no uncertainty bounds."""
        mean_pred = self.predict(X)
        return mean_pred, mean_pred, mean_pred

    def timed_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Wrapper that records training time."""
        start = time.perf_counter()
        self.fit(X_train, y_train, X_val, y_val, feature_names)
        self.training_time_s = time.perf_counter() - start
        self.is_fitted = True
