from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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
        self.feature_columns = [
            "day_of_year",
            "month_sin",
            "month_cos",
            "rainfall_lag_1",
            "rainfall_lag_7",
            "rainfall_roll_7",
            "tmax_lag_1",
            "tmax_roll_7",
            "tmin_lag_1",
            "humidity_lag_1",
            "diurnal_range_lag_1",
            "latitude",
            "longitude",
            "region_code",
        ]

    def prepare_features(self, observations: pd.DataFrame) -> pd.DataFrame:
        frame = observations.sort_values(["region", "date"]).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["day_of_year"] = frame["date"].dt.dayofyear
        frame["month"] = frame["date"].dt.month
        frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12.0)
        frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12.0)
        frame["region_code"] = frame["region"].astype("category").cat.codes

        if not self.region_code_map:
            ordered_regions = sorted(frame["region"].astype(str).unique())
            self.region_code_map = {region: index for index, region in enumerate(ordered_regions)}
        frame["region_code"] = frame["region"].map(self.region_code_map).fillna(0).astype(int)

        lag_groups = frame.groupby("region", group_keys=False)
        frame["rainfall_lag_1"] = lag_groups["rainfall_mm"].shift(1)
        frame["rainfall_lag_7"] = lag_groups["rainfall_mm"].shift(7)
        frame["rainfall_roll_7"] = lag_groups["rainfall_mm"].transform(lambda series: series.shift(1).rolling(7, min_periods=3).mean())
        frame["tmax_lag_1"] = lag_groups["tmax_c"].shift(1)
        frame["tmax_roll_7"] = lag_groups["tmax_c"].transform(lambda series: series.shift(1).rolling(7, min_periods=3).mean())
        frame["tmin_lag_1"] = lag_groups["tmin_c"].shift(1)
        frame["humidity_lag_1"] = lag_groups["humidity_pct"].shift(1)
        frame["diurnal_range_lag_1"] = lag_groups["diurnal_range_c"].shift(1) if "diurnal_range_c" in frame.columns else frame["tmax_lag_1"] - frame["tmin_lag_1"]

        frame = frame.dropna().reset_index(drop=True)
        return frame.drop(columns=["month"])

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
        sorted_history = history.sort_values("date").reset_index(drop=True)
        last_row = sorted_history.iloc[-1]
        region_code = self.region_code_map.get(str(last_row["region"]), 0)
        target_date = pd.to_datetime(last_row["date"]) + pd.Timedelta(days=step_ahead)
        month = target_date.month
        tail = sorted_history.tail(7)
        diurnal_tail = tail["diurnal_range_c"] if "diurnal_range_c" in tail.columns else tail["tmax_c"] - tail["tmin_c"]

        return pd.DataFrame(
            [
                {
                    "day_of_year": target_date.dayofyear,
                    "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
                    "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
                    "rainfall_lag_1": float(last_row["rainfall_mm"]),
                    "rainfall_lag_7": float(sorted_history.iloc[-7]["rainfall_mm"]) if len(sorted_history) >= 7 else float(last_row["rainfall_mm"]),
                    "rainfall_roll_7": float(tail["rainfall_mm"].mean()),
                    "tmax_lag_1": float(last_row["tmax_c"]),
                    "tmax_roll_7": float(tail["tmax_c"].mean()),
                    "tmin_lag_1": float(last_row["tmin_c"]),
                    "humidity_lag_1": float(last_row["humidity_pct"]),
                    "diurnal_range_lag_1": float(diurnal_tail.iloc[-1] if hasattr(diurnal_tail, "iloc") else diurnal_tail),
                    "latitude": float(last_row["latitude"]),
                    "longitude": float(last_row["longitude"]),
                    "region_code": int(region_code),
                }
            ]
        )
