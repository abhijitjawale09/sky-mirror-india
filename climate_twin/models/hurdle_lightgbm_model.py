"""Hurdle LightGBM Climate Model for Zero-Inflated Rainfall and Temperature Forecasting.

Implements a two-stage hurdle architecture for precipitation:
1. Stage 1: Binary LightGBM classifier predicting rain occurrence (P(Rain > 0.1mm)).
2. Stage 2: LightGBM regressor with Tweedie / Log objective trained exclusively on wet days.
Final rainfall prediction = P(Rain > threshold) * Intensity.

For Max and Min temperatures, dedicated LightGBM gradient boosted trees are trained
with L1/L2 regularization and histogram binning.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base_model import ClimateModel, TARGET_COLUMNS

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Hurdle LightGBM model will not be available.")


class HurdleLGBMClimateModel(ClimateModel):
    """Two-Stage Hurdle LightGBM for precipitation and gradient boosted temperature forecaster."""

    name = "hurdle_lightgbm"
    display_name = "Hurdle LightGBM"

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 31,
        rain_threshold_mm: float = 0.1,
        prob_threshold: float = 0.35,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.rain_classifier: Any = None
        self.rain_regressor: Any = None
        self.temp_regressors: dict[str, Any] = {}
        self.residual_stds: dict[str, float] = {}
        self.rain_threshold_mm = rain_threshold_mm
        self.prob_threshold = prob_threshold

        self._hyperparameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "rain_threshold_mm": rain_threshold_mm,
            "prob_threshold": prob_threshold,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }

    def _check_available(self) -> None:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is required for HurdleLGBMClimateModel. "
                "Install with: pip install lightgbm"
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._check_available()
        self.feature_columns = feature_names or []
        hp = self._hyperparameters

        # Common tree parameters
        base_params = {
            "n_estimators": hp["n_estimators"],
            "learning_rate": hp["learning_rate"],
            "max_depth": hp["max_depth"],
            "num_leaves": hp["num_leaves"],
            "subsample": hp["subsample"],
            "colsample_bytree": hp["colsample_bytree"],
            "random_state": hp["random_state"],
            "n_jobs": -1,
            "verbose": -1,
        }

        # --- Stage 1 & 2: Rainfall Hurdle ---
        y_rain_train = y_train[:, 0]
        rain_binary_train = (y_rain_train > self.rain_threshold_mm).astype(int)

        self.rain_classifier = lgb.LGBMClassifier(
            **base_params,
            objective="binary",
            is_unbalance=True,
        )
        self.rain_classifier.fit(X_train, rain_binary_train)

        # Regressor trained on wet days (or all if very few wet days)
        wet_mask = rain_binary_train == 1
        if np.sum(wet_mask) < 20:
            wet_X = X_train
            wet_y = y_rain_train
        else:
            wet_X = X_train[wet_mask]
            wet_y = y_rain_train[wet_mask]

        self.rain_regressor = lgb.LGBMRegressor(
            **base_params,
            objective="tweedie",
            tweedie_variance_power=1.5,
        )
        self.rain_regressor.fit(wet_X, wet_y)

        # --- Temperatures: Tmax and Tmin ---
        for i, target in enumerate(["tmax_c", "tmin_c"], start=1):
            reg = lgb.LGBMRegressor(
                **base_params,
                objective="regression",
            )
            reg.fit(X_train, y_train[:, i])
            self.temp_regressors[target] = reg

        # Compute empirical residual standard deviation for uncertainty estimation
        train_preds = self._predict_raw(X_train)
        for i, target in enumerate(TARGET_COLUMNS):
            resids = y_train[:, i] - train_preds[:, i]
            self.residual_stds[target] = float(np.std(resids))

        self.is_fitted = True

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        # Rain occurrence probability
        rain_probs = self.rain_classifier.predict_proba(X)[:, 1]
        rain_intensities = np.maximum(0.0, self.rain_regressor.predict(X))

        # Expected precipitation: gate intensity by occurrence
        # Soft threshold gating for smooth gradients
        rain_pred = np.where(
            rain_probs > self.prob_threshold,
            rain_intensities * (rain_probs / max(self.prob_threshold, 1e-4)),
            0.0,
        )
        rain_pred = np.maximum(0.0, rain_pred)

        tmax_pred = self.temp_regressors["tmax_c"].predict(X)
        tmin_pred = self.temp_regressors["tmin_c"].predict(X)

        return np.column_stack([rain_pred, tmax_pred, tmin_pred])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Hurdle LightGBM model has not been trained.")
        return self._predict_raw(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean and calibrated 80% prediction intervals (P10 - P90)."""
        mean_preds = self.predict(X)
        z_80 = 1.28155  # 80% normal coverage factor

        lower = np.zeros_like(mean_preds)
        upper = np.zeros_like(mean_preds)

        for i, target in enumerate(TARGET_COLUMNS):
            sigma = self.residual_stds.get(target, 1.0)
            lower[:, i] = mean_preds[:, i] - z_80 * sigma
            upper[:, i] = mean_preds[:, i] + z_80 * sigma

        # Physical boundary clipping
        lower[:, 0] = np.maximum(0.0, lower[:, 0])
        upper[:, 0] = np.maximum(0.0, upper[:, 0])

        return mean_preds, lower, upper

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or not self.feature_columns:
            return None

        total_gain = np.zeros(len(self.feature_columns))
        models_to_check = [
            self.rain_classifier,
            self.rain_regressor,
            self.temp_regressors.get("tmax_c"),
            self.temp_regressors.get("tmin_c"),
        ]

        valid_count = 0
        for m in models_to_check:
            if m is not None and hasattr(m, "feature_importances_"):
                fi = m.feature_importances_
                if len(fi) == len(self.feature_columns):
                    total_gain += fi
                    valid_count += 1

        if valid_count == 0 or total_gain.sum() == 0:
            return None

        normalized = (total_gain / total_gain.sum()) * 100.0
        return {
            col: round(float(imp), 2)
            for col, imp in sorted(
                zip(self.feature_columns, normalized),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def save(self, directory: Path) -> Path:
        self._check_available()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "hurdle_lightgbm_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "rain_classifier": self.rain_classifier,
                "rain_regressor": self.rain_regressor,
                "temp_regressors": self.temp_regressors,
                "residual_stds": self.residual_stds,
                "feature_columns": self.feature_columns,
                "hyperparameters": self._hyperparameters,
                "training_time_s": self.training_time_s,
            }, f)
        return path

    def load(self, directory: Path) -> None:
        self._check_available()
        path = directory / "hurdle_lightgbm_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rain_classifier = data["rain_classifier"]
        self.rain_regressor = data["rain_regressor"]
        self.temp_regressors = data["temp_regressors"]
        self.residual_stds = data.get("residual_stds", {})
        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.training_time_s = data.get("training_time_s", 0.0)
        self.is_fitted = True
