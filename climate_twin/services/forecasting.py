from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
    rainfall_confidence: float
    tmax_confidence: float
    tmin_confidence: float


class ClimateForecaster:
    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None
        self.region_code_map: dict[str, int] = {}
        self.feature_columns = FEATURE_COLUMNS.copy()

    def prepare_features(self, observations: pd.DataFrame) -> pd.DataFrame:
        feature_frame, self.region_code_map = prepare_training_features(observations)
        return feature_frame

    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        frame = self.prepare_features(observations)
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
        self.model.fit(train_frame[self.feature_columns], train_frame[["rainfall_mm", "tmax_c", "tmin_c"]])

        metrics: dict[str, float] = {}
        if not test_frame.empty:
            predictions = self.model.predict(test_frame[self.feature_columns])
            metrics["mae_rainfall_mm"] = float(mean_absolute_error(test_frame["rainfall_mm"], predictions[:, 0]))
            metrics["mae_tmax_c"] = float(mean_absolute_error(test_frame["tmax_c"], predictions[:, 1]))
            metrics["mae_tmin_c"] = float(mean_absolute_error(test_frame["tmin_c"], predictions[:, 2]))

        return metrics

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
            prediction = self.model.predict(features[self.feature_columns])[0]

            rainfall = max(0.0, float(prediction[0]) * (1.0 + rainfall_delta_pct / 100.0))
            tmax = float(prediction[1]) + temp_delta_c
            tmin = float(prediction[2]) + temp_delta_c * 0.82

            forecast_date = pd.to_datetime(working_history.iloc[-1]["date"]) + pd.Timedelta(days=1)
            next_row = {
                "date": forecast_date,
                "region": region,
                "rainfall_mm": round(rainfall, 2),
                "tmax_c": round(tmax, 2),
                "tmin_c": round(tmin, 2),
                "latitude": float(working_history.iloc[-1]["latitude"]),
                "longitude": float(working_history.iloc[-1]["longitude"]),
                "humidity_pct": max(35.0, min(98.0, float(working_history.iloc[-1]["humidity_pct"]) + rainfall_delta_pct * 0.12 - temp_delta_c * 0.8)),
                "diurnal_range_c": max(1.0, tmax - tmin),
            }
            forecast_rows.append({key: next_row[key] for key in ["date", "region", "rainfall_mm", "tmax_c", "tmin_c"]})

            working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

        return pd.DataFrame(forecast_rows)

    def _make_feature_row(self, history: pd.DataFrame, step_ahead: int) -> pd.DataFrame:
        target_date = pd.to_datetime(history.iloc[-1]["date"]) + pd.Timedelta(days=step_ahead)
        return prepare_inference_features(
            history=history,
            region_code_map=self.region_code_map,
            target_date=target_date,
            step_ahead=step_ahead,
        )
