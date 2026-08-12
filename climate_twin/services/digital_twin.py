from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.loader import REGION_PROFILES, get_region_profile, latest_snapshot, load_training_observations
from .forecasting import ClimateForecaster
from .twin_simulation import TwinScenario, build_replay_result, build_simulation_result
from .twin_state import build_twin_state


@dataclass
class DashboardMode:
    """Describe how the dashboard payload should be assembled."""

    name: str = "live"
    replay_start: str | None = None
    replay_end: str | None = None


class DigitalTwinEngine:
    """Orchestrate live state, what-if simulation, and replay reporting."""

    def __init__(self) -> None:
        self.observations, self.data_source = load_training_observations()
        self.forecaster = ClimateForecaster()
        self.metrics = self.forecaster.fit(self.observations)

    def get_dashboard_state(
        self,
        region_name: str,
        scenario: TwinScenario | None = None,
        mode: DashboardMode | None = None,
    ) -> dict:
        """Return a JSON-serializable dashboard state without breaking legacy keys."""

        mode = mode or DashboardMode()
        scenario = scenario or TwinScenario()
        profile = get_region_profile(region_name)
        region_history = self.observations.loc[self.observations["region"] == profile.name].sort_values("date")
        snapshot = latest_snapshot(self.observations, profile.name)
        twin_state = build_twin_state(region_history)
        forecast = self.forecaster.predict_next(
            region_history,
            horizon_days=scenario.horizon_days,
            rainfall_delta_pct=scenario.rainfall_delta_pct,
            temp_delta_c=scenario.temp_delta_c,
        )
        simulation = build_simulation_result(self.forecaster, region_history, scenario)
        replay = None
        if mode.name == "replay" and mode.replay_start and mode.replay_end:
            replay = build_replay_result(self.forecaster, region_history, mode.replay_start, mode.replay_end)

        return {
            "region": profile.name,
            "mode": mode.name,
            "snapshot": snapshot,
            "snapshot_basis": {
                "rainfall_mm": "observed",
                "tmax_c": "observed",
                "tmin_c": "observed",
                "humidity_pct": "observed",
                "rainfall_anomaly": "rule_derived",
                "temperature_anomaly": "rule_derived",
                "latitude": "observed",
                "longitude": "observed",
            },
            "twin_state": twin_state.to_dict(),
            "forecast": self._forecast_payload(forecast),
            "forecast_basis": {
                "labels": "ml_forecast",
                "rainfall": "ml_forecast",
                "tmax": "ml_forecast",
                "tmin": "ml_forecast",
            },
            "time_series": self._build_time_series(region_history, forecast),
            "time_series_basis": {"observed": "observed", "forecast": "ml_forecast"},
            "anomalies": self._build_anomaly_series(region_history, forecast),
            "anomaly_basis": "rule_derived",
            "risk_level": twin_state.risk_level,
            "metrics": self._summary_metrics(region_history, forecast, scenario, twin_state.risk_level),
            "metrics_basis": {
                "monsoon_pulse": "rule_derived",
                "heat_stress": "rule_derived",
                "stability": "rule_derived",
                "predicted_rainfall": "ml_forecast",
                "predicted_tmax": "ml_forecast",
                "risk_level": "rule_derived",
            },
            "simulation": simulation.to_dict(),
            "replay": replay.to_dict() if replay is not None else None,
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

    def _build_time_series(self, region_history: pd.DataFrame, forecast: pd.DataFrame) -> dict:
        history = region_history.sort_values("date").tail(30).copy()
        history["date"] = pd.to_datetime(history["date"])
        forecast_frame = forecast.copy()
        forecast_frame["date"] = pd.to_datetime(forecast_frame["date"])
        return {
            "labels": history["date"].dt.strftime("%d %b").tolist() + forecast_frame["date"].dt.strftime("%d %b").tolist(),
            "observed": {
                "rainfall_mm": history["rainfall_mm"].round(2).tolist(),
                "tmax_c": history["tmax_c"].round(2).tolist(),
                "tmin_c": history["tmin_c"].round(2).tolist(),
            },
            "forecast": {
                "rainfall_mm": forecast_frame["rainfall_mm"].round(2).tolist(),
                "tmax_c": forecast_frame["tmax_c"].round(2).tolist(),
                "tmin_c": forecast_frame["tmin_c"].round(2).tolist(),
            },
            "basis": {"observed": "observed", "forecast": "ml_forecast"},
        }

    def _build_anomaly_series(self, region_history: pd.DataFrame, forecast: pd.DataFrame) -> dict:
        history = region_history.sort_values("date").tail(30).copy()
        history["date"] = pd.to_datetime(history["date"])
        forecast_frame = forecast.copy()
        forecast_frame["date"] = pd.to_datetime(forecast_frame["date"])

        history_rainfall_anomalies = []
        history_tmax_anomalies = []
        history_tmin_anomalies = []
        for row in history.itertuples(index=False):
            baseline = self._same_day_normal(region_history, pd.to_datetime(row.date))
            history_rainfall_anomalies.append(float(row.rainfall_mm - baseline["rainfall_mm"]))
            history_tmax_anomalies.append(float(row.tmax_c - baseline["tmax_c"]))
            history_tmin_anomalies.append(float(row.tmin_c - baseline["tmin_c"]))

        forecast_rainfall_anomalies = []
        forecast_tmax_anomalies = []
        forecast_tmin_anomalies = []
        for row in forecast_frame.itertuples(index=False):
            baseline = self._same_day_normal(region_history, pd.to_datetime(row.date))
            forecast_rainfall_anomalies.append(float(row.rainfall_mm - baseline["rainfall_mm"]))
            forecast_tmax_anomalies.append(float(row.tmax_c - baseline["tmax_c"]))
            forecast_tmin_anomalies.append(float(row.tmin_c - baseline["tmin_c"]))

        return {
            "labels": history["date"].dt.strftime("%d %b").tolist() + forecast_frame["date"].dt.strftime("%d %b").tolist(),
            "rainfall_mm": history_rainfall_anomalies + forecast_rainfall_anomalies,
            "tmax_c": history_tmax_anomalies + forecast_tmax_anomalies,
            "tmin_c": history_tmin_anomalies + forecast_tmin_anomalies,
            "basis": "rule_derived",
        }

    def _same_day_normal(self, region_history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
        history = region_history.copy()
        history["date"] = pd.to_datetime(history["date"])
        day_of_year = int(pd.to_datetime(target_date).dayofyear)
        same_day = history.loc[history["date"].dt.dayofyear == day_of_year]
        if same_day.empty:
            same_day = history
        return {
            "rainfall_mm": float(same_day["rainfall_mm"].mean()),
            "tmax_c": float(same_day["tmax_c"].mean()),
            "tmin_c": float(same_day["tmin_c"].mean()),
        }

    def _summary_metrics(self, region_history: pd.DataFrame, forecast: pd.DataFrame, scenario: TwinScenario, risk_level: str) -> dict:
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
            "risk_level": risk_level,
        }

    def _source_manifest(self) -> list[dict[str, str]]:
        return [
            {
                    "name": "IMD 15-Year Fusion",
                "source": "data/processed/climate_training_data.csv",
                "role": "cached training table built from 15 years of rainfall, max temp, and min temp CSVs",
            },
            {"name": "IMD Gridded Rainfall", "source": "imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html", "role": "target variable and validation"},
            {"name": "IMD Maximum Temperature", "source": "imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html", "role": "thermal profile modeling"},
            {"name": "IMD Minimum Temperature", "source": "imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html", "role": "night-time thermal regime"},
            {"name": "INSAT / MOSDAC Layer", "source": "mosdac.gov.in", "role": "satellite augmentation and spatial context"},
        ]
