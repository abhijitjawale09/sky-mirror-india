"""Physics-Guided Stacking Meta-Ensemble for Climate Prediction.

Combines heterogeneous Level-0 estimators (Random Forest, HistGradientBoosting,
XGBoost, and LightGBM) via a non-negative regularized meta-regressor (Ridge).

Applies atmospheric physics consistency constraints:
1. Non-negative precipitation: Rain >= 0.0 mm.
2. Thermodynamic diurnal inequality: Tmax >= Tmin + 0.5 °C (conserving thermal energy).
3. Physics-weighted prediction variance for calibrated 80% confidence bounds.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .base_model import ClimateModel, TARGET_COLUMNS

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class PhysicsStackingClimateModel(ClimateModel):
    """Stacking ensemble with non-negative meta-weights and physical law constraints."""

    name = "stacking_ensemble"
    display_name = "Physics Stacking Ensemble"

    def __init__(
        self,
        n_cv_folds: int = 5,
        meta_alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_cv_folds = n_cv_folds
        self.meta_alpha = meta_alpha
        self.random_state = random_state

        # Base estimators per target: {target: [model1, model2, ...]}
        self.base_models: dict[str, list[Any]] = {t: [] for t in TARGET_COLUMNS}
        # Meta learners per target
        self.meta_learners: dict[str, Ridge] = {}
        self.meta_weights: dict[str, np.ndarray] = {}
        self.residual_stds: dict[str, float] = {}

        self._hyperparameters = {
            "n_cv_folds": n_cv_folds,
            "meta_alpha": meta_alpha,
            "random_state": random_state,
        }

    def _init_base_estimators(self) -> list[tuple[str, Any]]:
        """Instantiate a set of diverse tree estimators."""
        estimators = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=16,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=self.random_state,
                ),
            ),
            (
                "hist_gb",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.05,
                    max_depth=10,
                    random_state=self.random_state,
                ),
            ),
        ]

        if XGB_AVAILABLE:
            estimators.append((
                "xgb",
                XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=self.random_state,
                ),
            ))

        if LGB_AVAILABLE:
            estimators.append((
                "lgb",
                lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    verbose=-1,
                    random_state=self.random_state,
                ),
            ))

        return estimators

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_columns = feature_names or []
        kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)

        for target_idx, target in enumerate(TARGET_COLUMNS):
            y_target = y_train[:, target_idx]
            base_specs = self._init_base_estimators()
            n_base = len(base_specs)

            # Generate out-of-fold predictions for level-1 training
            oof_preds = np.zeros((len(X_train), n_base))

            for train_idx, val_idx in kf.split(X_train):
                fold_X_tr, fold_y_tr = X_train[train_idx], y_target[train_idx]
                fold_X_va = X_train[val_idx]

                for b_idx, (_, estimator) in enumerate(self._init_base_estimators()):
                    estimator.fit(fold_X_tr, fold_y_tr)
                    oof_preds[val_idx, b_idx] = estimator.predict(fold_X_va)

            # Fit Level-1 Meta-Learner with non-negative regularized weights
            meta = Ridge(alpha=self.meta_alpha, positive=True, fit_intercept=True)
            meta.fit(oof_preds, y_target)
            self.meta_learners[target] = meta
            self.meta_weights[target] = meta.coef_

            # Refit all base models on the entire training set
            full_base_models = []
            for _, estimator in base_specs:
                estimator.fit(X_train, y_target)
                full_base_models.append(estimator)
            self.base_models[target] = full_base_models

        # Compute empirical residual standard deviations
        train_preds = self._predict_unconstrained(X_train)
        train_preds = self._apply_physics_constraints(train_preds)
        for i, target in enumerate(TARGET_COLUMNS):
            diff = y_train[:, i] - train_preds[:, i]
            self.residual_stds[target] = float(np.std(diff))

        self.is_fitted = True

    def _predict_unconstrained(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X), len(TARGET_COLUMNS)))
        for i, target in enumerate(TARGET_COLUMNS):
            base_preds = np.column_stack([
                model.predict(X) for model in self.base_models[target]
            ])
            preds[:, i] = self.meta_learners[target].predict(base_preds)
        return preds

    def _apply_physics_constraints(self, preds: np.ndarray) -> np.ndarray:
        """Enforce fundamental atmospheric physical boundaries."""
        constrained = preds.copy()

        # 1. Non-negative precipitation
        constrained[:, 0] = np.maximum(0.0, constrained[:, 0])

        # 2. Thermodynamic constraint: Tmax >= Tmin + 0.5 °C
        # If violated, redistribute around midpoint to conserve thermal budget
        tmax = constrained[:, 1]
        tmin = constrained[:, 2]
        violation_mask = tmax < (tmin + 0.5)

        if np.any(violation_mask):
            t_mid = (tmax[violation_mask] + tmin[violation_mask]) / 2.0
            constrained[violation_mask, 1] = t_mid + 0.25
            constrained[violation_mask, 2] = t_mid - 0.25

        return constrained

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Physics Stacking Ensemble has not been trained.")
        unconstrained = self._predict_unconstrained(X)
        return self._apply_physics_constraints(unconstrained)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blend base-model spread and residual variance for calibrated 80% CI."""
        mean_preds = self.predict(X)
        z_80 = 1.28155

        lower = np.zeros_like(mean_preds)
        upper = np.zeros_like(mean_preds)

        for i, target in enumerate(TARGET_COLUMNS):
            # Model diversity spread
            base_preds = np.column_stack([
                model.predict(X) for model in self.base_models[target]
            ])
            ensemble_std = np.std(base_preds, axis=1)
            residual_sigma = self.residual_stds.get(target, 1.0)
            combined_sigma = np.sqrt(ensemble_std**2 + residual_sigma**2)

            lower[:, i] = mean_preds[:, i] - z_80 * combined_sigma
            upper[:, i] = mean_preds[:, i] + z_80 * combined_sigma

        lower[:, 0] = np.maximum(0.0, lower[:, 0])
        upper[:, 0] = np.maximum(0.0, upper[:, 0])

        # Enforce physics on bounds
        lower = self._apply_physics_constraints(lower)
        upper = self._apply_physics_constraints(upper)

        return mean_preds, lower, upper

    def get_feature_importance(self) -> dict[str, float] | None:
        """Compute meta-weighted feature importances from base tree models."""
        if not self.is_fitted or not self.feature_columns:
            return None

        combined_importances = np.zeros(len(self.feature_columns))

        for target in TARGET_COLUMNS:
            models = self.base_models.get(target, [])
            weights = self.meta_weights.get(target, np.ones(len(models)))
            weight_sum = np.sum(weights) if np.sum(weights) > 0 else 1.0
            norm_weights = weights / weight_sum

            for model, w in zip(models, norm_weights):
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    if len(fi) == len(self.feature_columns):
                        combined_importances += w * fi

        total = np.sum(combined_importances)
        if total == 0:
            return None

        normalized = (combined_importances / total) * 100.0
        return {
            col: round(float(imp), 2)
            for col, imp in sorted(
                zip(self.feature_columns, normalized),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "stacking_ensemble.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "base_models": self.base_models,
                "meta_learners": self.meta_learners,
                "meta_weights": self.meta_weights,
                "residual_stds": self.residual_stds,
                "feature_columns": self.feature_columns,
                "hyperparameters": self._hyperparameters,
                "training_time_s": self.training_time_s,
            }, f)
        return path

    def load(self, directory: Path) -> None:
        path = directory / "stacking_ensemble.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.base_models = data["base_models"]
        self.meta_learners = data["meta_learners"]
        self.meta_weights = data["meta_weights"]
        self.residual_stds = data.get("residual_stds", {})
        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.training_time_s = data.get("training_time_s", 0.0)
        self.is_fitted = True
