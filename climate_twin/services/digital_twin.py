from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.loader import REGION_PROFILES, get_region_profile, latest_snapshot, load_training_observations
from .forecasting import ClimateForecaster


@dataclass
class TwinScenario:
    rainfall_delta_pct: float = 0.0
    temp_delta_c: float = 0.0
    horizon_days: int = 14


class DigitalTwinEngine:
    def __init__(self) -> None:
        self.observations, self.data_source = load_training_observations()
        self.forecaster = ClimateForecaster()
        self.metrics = self.forecaster.fit(self.observations)

    def get_dashboard_state(self, region_name: str, scenario: TwinScenario | None = None) -> dict:
        scenario = scenario or TwinScenario()
        profile = get_region_profile(region_name)
        region_history = self.observations.loc[self.observations["region"] == profile.name].sort_values("date")
        snapshot = latest_snapshot(self.observations, profile.name)
        forecast = self.forecaster.predict_next(
            region_history,
            horizon_days=scenario.horizon_days,
            rainfall_delta_pct=scenario.rainfall_delta_pct,
            temp_delta_c=scenario.temp_delta_c,
        )

        return {
            "region": profile.name,
            "snapshot": snapshot,
            "forecast": self._forecast_payload(forecast),
            "metrics": self._summary_metrics(region_history, forecast, scenario),
            "model_metrics": self.metrics,
            "data_source": self.data_source,
            "source_manifest": self._source_manifest(),
            "pilot_regions": [
                {"name": region.name, "latitude": region.latitude, "longitude": region.longitude}
                for region in REGION_PROFILES
            ],
        }

    def _forecast_payload(self, forecast: pd.DataFrame) -> dict:
        forecast = forecast.copy()
        forecast["date"] = pd.to_datetime(forecast["date"])
        return {
            "labels": forecast["date"].dt.strftime("%d %b").tolist(),
            "rainfall": forecast["rainfall_mm"].round(2).tolist(),
            "tmax": forecast["tmax_c"].round(2).tolist(),
            "tmin": forecast["tmin_c"].round(2).tolist(),
        }

    def _summary_metrics(self, region_history: pd.DataFrame, forecast: pd.DataFrame, scenario: TwinScenario) -> dict:
        recent = region_history.tail(14)
        avg_rainfall = float(recent["rainfall_mm"].mean())
        avg_tmax = float(recent["tmax_c"].mean())
        predicted_rainfall = float(forecast["rainfall_mm"].mean())
        predicted_tmax = float(forecast["tmax_c"].mean())

        monsoon_pulse = max(0.0, min(100.0, predicted_rainfall * 12 + scenario.rainfall_delta_pct * 0.9))
        heat_stress = max(0.0, min(100.0, (predicted_tmax - 26.0) * 4 + scenario.temp_delta_c * 8))
        stability = max(0.0, min(100.0, 100.0 - abs(predicted_rainfall - avg_rainfall) * 5 - abs(predicted_tmax - avg_tmax) * 3))

        return {
            "monsoon_pulse": round(monsoon_pulse, 1),
            "heat_stress": round(heat_stress, 1),
            "stability": round(stability, 1),
            "predicted_rainfall": round(predicted_rainfall, 2),
            "predicted_tmax": round(predicted_tmax, 2),
        }

    def _source_manifest(self) -> list[dict[str, str]]:
        return [
            {
                "name": "Processed IMD Fusion",
                "source": "data/processed/climate_training_data.csv",
                "role": "cached training table built from rainfall, max temp, and min temp CSVs",
            },
            {"name": "IMD Gridded Rainfall", "source": "imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html", "role": "target variable and validation"},
            {"name": "IMD Maximum Temperature", "source": "imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html", "role": "thermal profile modeling"},
            {"name": "IMD Minimum Temperature", "source": "imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html", "role": "night-time thermal regime"},
            {"name": "INSAT / MOSDAC Layer", "source": "mosdac.gov.in", "role": "satellite augmentation and spatial context"},
        ]
