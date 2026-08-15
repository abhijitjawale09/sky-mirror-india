from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    prepare_inference_features,
    prepare_training_features,
)


@dataclass
class ForecastBundle:
    rainfall: float
    tmax: float
    tmin: float
    rainfall_lower: float
    rainfall_upper: float
    tmax_lower: float
    tmax_upper: float
    tmin_lower: float
    tmin_upper: float


class ClimateForecaster:
    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None
        self.region_code_map: dict[str, int] = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.feature_importance: dict[str, float] = {}
        self.baseline_metrics: dict[str, dict[str, float]] = {}
        self.deterministic_metrics: dict[str, dict[str, float]] = {}

    def prepare_features(self, observations: pd.DataFrame) -> pd.DataFrame:
        feature_frame, self.region_code_map = prepare_training_features(observations)
        return feature_frame

    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        frame = self.prepare_features(observations)

        # Temporal split: use last 15% as test set (approximates last year)
        split_index = max(30, int(len(frame) * 0.85))
        train_frame = frame.iloc[:split_index]
        test_frame = frame.iloc[split_index:]

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(train_frame[self.feature_columns], train_frame[TARGET_COLUMNS])

        # Extract feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            col: round(float(imp) * 100, 1)
            for col, imp in sorted(
                zip(self.feature_columns, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        metrics: dict[str, float] = {}
        if not test_frame.empty:
            predictions = self.model.predict(test_frame[self.feature_columns])
            y_test = test_frame[TARGET_COLUMNS].values

            # Compute baseline metrics (persistence model = yesterday's value)
            persistence = test_frame[["rainfall_lag_1", "tmax_lag_1", "tmin_lag_1"]].values

            target_names = ["rainfall_mm", "tmax_c", "tmin_c"]
            persistence_cols = ["rainfall_lag_1", "tmax_lag_1", "tmin_lag_1"]

            for i, (name, pcol) in enumerate(zip(target_names, persistence_cols)):
                # RF metrics
                mae = float(mean_absolute_error(y_test[:, i], predictions[:, i]))
                rmse = float(np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])))
                r2 = float(r2_score(y_test[:, i], predictions[:, i]))
                metrics[f"mae_{name}"] = mae

                self.deterministic_metrics[name] = {
                    "mae": round(mae, 3),
                    "rmse": round(rmse, 3),
                    "r2": round(r2, 3),
                }

                # Persistence baseline metrics
                p_vals = test_frame[pcol].values
                p_mae = float(mean_absolute_error(y_test[:, i], p_vals))
                p_rmse = float(np.sqrt(mean_squared_error(y_test[:, i], p_vals)))
                p_r2 = float(r2_score(y_test[:, i], p_vals))
                self.baseline_metrics[name] = {
                    "mae": round(p_mae, 3),
                    "rmse": round(p_rmse, 3),
                    "r2": round(p_r2, 3),
                }

        return metrics

    def predict_with_uncertainty(
        self,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict using all trees individually and return mean, lower, upper (80% interval)."""
        if self.model is None:
            raise RuntimeError("Forecast model has not been trained")

        # Get predictions from each individual tree
        tree_predictions = np.array([
            tree.predict(features[self.feature_columns].values)
            for tree in self.model.estimators_
        ])  # shape: (n_trees, n_samples, 3)

        mean_pred = np.mean(tree_predictions, axis=0)
        lower = np.percentile(tree_predictions, 10, axis=0)
        upper = np.percentile(tree_predictions, 90, axis=0)

        return mean_pred, lower, upper

    def predict_next(
        self,
        history: pd.DataFrame,
        horizon_days: int = 14,
        rainfall_delta_pct: float = 0.0,
        temp_delta_c: float = 0.0,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Forecast model has not been trained")

        region = str(history.iloc[-1]["region"])
        forecast_rows: list[dict[str, float | str | pd.Timestamp]] = []
        working_history = history.copy().sort_values("date").reset_index(drop=True)

        for step in range(horizon_days):
            features = self._make_feature_row(working_history, step + 1)
            mean_pred, lower, upper = self.predict_with_uncertainty(features)

            rainfall = max(0.0, float(mean_pred[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            tmax = float(mean_pred[0][1]) + temp_delta_c
            tmin = float(mean_pred[0][2]) + temp_delta_c * 0.82

            # Uncertainty bounds with scenario adjustments
            rain_lower = max(0.0, float(lower[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            rain_upper = max(0.0, float(upper[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            tmax_lower = float(lower[0][1]) + temp_delta_c
            tmax_upper = float(upper[0][1]) + temp_delta_c
            tmin_lower = float(lower[0][2]) + temp_delta_c * 0.82
            tmin_upper = float(upper[0][2]) + temp_delta_c * 0.82

            forecast_date = pd.to_datetime(working_history.iloc[-1]["date"]) + pd.Timedelta(days=1)
            next_row = {
                "date": forecast_date,
                "region": region,
                "rainfall_mm": round(rainfall, 2),
                "tmax_c": round(tmax, 2),
                "tmin_c": round(tmin, 2),
                "rainfall_lower": round(rain_lower, 2),
                "rainfall_upper": round(rain_upper, 2),
                "tmax_lower": round(tmax_lower, 2),
                "tmax_upper": round(tmax_upper, 2),
                "tmin_lower": round(tmin_lower, 2),
                "tmin_upper": round(tmin_upper, 2),
                "latitude": float(working_history.iloc[-1]["latitude"]),
                "longitude": float(working_history.iloc[-1]["longitude"]),
                "humidity_pct": max(35.0, min(98.0, float(working_history.iloc[-1]["humidity_pct"]) + rainfall_delta_pct * 0.12 - temp_delta_c * 0.8)),
                "diurnal_range_c": max(1.0, tmax - tmin),
            }
            forecast_rows.append(next_row)

            working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

        return pd.DataFrame(forecast_rows)

    def get_model_comparison(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return a comparison table of baseline vs RF model."""
        return {
            "persistence": self.baseline_metrics,
            "random_forest": self.deterministic_metrics,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances as percentages, sorted descending."""
        return self.feature_importance

    def _make_feature_row(self, history: pd.DataFrame, step_ahead: int) -> pd.DataFrame:
        target_date = pd.to_datetime(history.iloc[-1]["date"]) + pd.Timedelta(days=step_ahead)
        return prepare_inference_features(
            history=history,
            region_code_map=self.region_code_map,
            target_date=target_date,
            step_ahead=step_ahead,
        )
