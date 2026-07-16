from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ForecastBundle:
    rainfall: float
    tmax: float
    tmin: float
    rainfall_confidence: float
    tmax_confidence: float
    tmin_confidence: float


class DigitalTwinNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 3) # Predicts rainfall, tmax, tmin
        )
        
    def forward(self, x):
        return self.net(x)


class ClimateForecaster:
    def __init__(self) -> None:
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.feature_columns = [
            "day_of_year",
            "rainfall_lag_1",
            "rainfall_lag_7",
            "rainfall_roll_7",
            "tmax_lag_1",
            "tmax_roll_7",
            "tmin_lag_1",
            "humidity_lag_1",
            "latitude",
            "longitude",
            "region_code",
        ]
        self.target_columns = ["rainfall_mm", "tmax_c", "tmin_c"]

    def prepare_features(self, observations: pd.DataFrame) -> pd.DataFrame:
        frame = observations.sort_values(["region", "date"]).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["day_of_year"] = frame["date"].dt.dayofyear
        frame["region_code"] = frame["region"].astype("category").cat.codes

        lag_groups = frame.groupby("region", group_keys=False)
        frame["rainfall_lag_1"] = lag_groups["rainfall_mm"].shift(1)
        frame["rainfall_lag_7"] = lag_groups["rainfall_mm"].shift(7)
        frame["rainfall_roll_7"] = lag_groups["rainfall_mm"].transform(lambda series: series.shift(1).rolling(7, min_periods=3).mean())
        frame["tmax_lag_1"] = lag_groups["tmax_c"].shift(1)
        frame["tmax_roll_7"] = lag_groups["tmax_c"].transform(lambda series: series.shift(1).rolling(7, min_periods=3).mean())
        frame["tmin_lag_1"] = lag_groups["tmin_c"].shift(1)
        frame["humidity_lag_1"] = lag_groups["humidity_pct"].shift(1)
        frame = frame.dropna().reset_index(drop=True)
        return frame

    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        frame = self.prepare_features(observations)
        metrics: dict[str, float] = {}
        split_index = max(30, int(len(frame) * 0.85))
        train_frame = frame.iloc[:split_index]
        test_frame = frame.iloc[split_index:]

        # Prepare tensors
        X_train = self.scaler_x.fit_transform(train_frame[self.feature_columns])
        y_train = self.scaler_y.fit_transform(train_frame[self.target_columns])
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        self.model = DigitalTwinNet(input_dim=len(self.feature_columns))
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Training loop
        self.model.train()
        epochs = 30
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation
        self.model.eval()
        if not test_frame.empty:
            X_test = self.scaler_x.transform(test_frame[self.feature_columns])
            with torch.no_grad():
                preds_scaled = self.model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                preds = self.scaler_y.inverse_transform(preds_scaled)
                
            metrics["mae_rainfall_mm"] = float(mean_absolute_error(test_frame["rainfall_mm"], preds[:, 0]))
            metrics["mae_tmax_c"] = float(mean_absolute_error(test_frame["tmax_c"], preds[:, 1]))
            metrics["mae_tmin_c"] = float(mean_absolute_error(test_frame["tmin_c"], preds[:, 2]))

        return metrics

    def predict_next(self, history: pd.DataFrame, horizon_days: int = 14, rainfall_delta_pct: float = 0.0, temp_delta_c: float = 0.0) -> pd.DataFrame:
        region = history.iloc[-1]["region"]
        forecast_rows = []
        working_history = history.copy().sort_values("date").reset_index(drop=True)

        self.model.eval()
        
        for step in range(horizon_days):
            features = self._make_feature_row(working_history, step + 1)
            X_feat = self.scaler_x.transform(features[self.feature_columns])
            
            with torch.no_grad():
                pred_scaled = self.model(torch.tensor(X_feat, dtype=torch.float32)).numpy()
                pred = self.scaler_y.inverse_transform(pred_scaled)[0]
                
            rainfall = float(pred[0])
            tmax = float(pred[1])
            tmin = float(pred[2])

            rainfall = max(0.0, rainfall * (1.0 + rainfall_delta_pct / 100.0))
            tmax = tmax + temp_delta_c
            tmin = tmin + temp_delta_c * 0.82

            forecast_date = pd.to_datetime(working_history.iloc[-1]["date"]) + pd.Timedelta(days=1)
            forecast_rows.append(
                {
                    "date": forecast_date,
                    "region": region,
                    "rainfall_mm": round(rainfall, 2),
                    "tmax_c": round(tmax, 2),
                    "tmin_c": round(tmin, 2),
                }
            )

            working_history = pd.concat(
                [
                    working_history,
                    pd.DataFrame(
                        [
                            forecast_rows[-1] | {
                                "latitude": working_history.iloc[-1]["latitude"],
                                "longitude": working_history.iloc[-1]["longitude"],
                                "humidity_pct": max(35.0, min(98.0, working_history.iloc[-1]["humidity_pct"] + rainfall_delta_pct * 0.12 - temp_delta_c * 0.8)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        return pd.DataFrame(forecast_rows)

    def _make_feature_row(self, history: pd.DataFrame, step_ahead: int) -> pd.DataFrame:
        sorted_history = history.sort_values("date").reset_index(drop=True)
        last_row = sorted_history.iloc[-1]
        region_code = sorted_history["region"].astype("category").cat.codes.iloc[-1]
        target_date = pd.to_datetime(last_row["date"]) + pd.Timedelta(days=step_ahead)
        tail = sorted_history.tail(7)

        return pd.DataFrame(
            [
                {
                    "day_of_year": target_date.dayofyear,
                    "rainfall_lag_1": float(last_row["rainfall_mm"]),
                    "rainfall_lag_7": float(sorted_history.iloc[-7]["rainfall_mm"]) if len(sorted_history) >= 7 else float(last_row["rainfall_mm"]),
                    "rainfall_roll_7": float(tail["rainfall_mm"].mean()),
                    "tmax_lag_1": float(last_row["tmax_c"]),
                    "tmax_roll_7": float(tail["tmax_c"].mean()),
                    "tmin_lag_1": float(last_row["tmin_c"]),
                    "humidity_lag_1": float(last_row["humidity_pct"]),
                    "latitude": float(last_row["latitude"]),
                    "longitude": float(last_row["longitude"]),
                    "region_code": int(region_code) if not pd.isna(region_code) else 0,
                }
            ]
        )
