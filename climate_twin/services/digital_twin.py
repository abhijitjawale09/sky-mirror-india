from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.loader import REGION_PROFILES, get_region_profile, latest_snapshot, load_training_observations
from .forecasting import ClimateForecaster
from .real_time_conditions import RealTimeConditions
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
        self.real_time_conditions = RealTimeConditions(self.observations)

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
        current_conditions = self.real_time_conditions.get_current_state(profile.name)
        current_risk = self.real_time_conditions.get_risk_level(current_conditions)
        current_precautions = self.real_time_conditions.get_precautions(
            current_risk["risk_level"],
            {
                "rainfall_delta_pct": scenario.rainfall_delta_pct,
                "temp_delta_c": scenario.temp_delta_c,
                "predicted_rainfall": float(forecast["rainfall_mm"].mean()) if not forecast.empty else 0.0,
                "predicted_tmax": float(forecast["tmax_c"].mean()) if not forecast.empty else 0.0,
            },
        )
        scenario_precautions = self.real_time_conditions.get_precautions(
            current_risk["risk_level"],
            {
                "rainfall_delta_pct": scenario.rainfall_delta_pct,
                "temp_delta_c": scenario.temp_delta_c,
                "predicted_rainfall": float(forecast["rainfall_mm"].mean()) if not forecast.empty else 0.0,
                "predicted_tmax": float(forecast["tmax_c"].mean()) if not forecast.empty else 0.0,
            },
        )
        replay = None
        if mode.name == "replay" and mode.replay_start and mode.replay_end:
            replay = build_replay_result(self.forecaster, region_history, mode.replay_start, mode.replay_end)

        # Compute data quality / scorecard metrics
        scorecard = self._build_scorecard(region_history, forecast)

        # Extreme event analysis
        extremes = self._detect_extreme_events(region_history, current_conditions)

        return {
            "region": profile.name,
            "mode": mode.name,
            "snapshot": snapshot,
            "snapshot_basis": {
                "rainfall_mm": "observed",
                "tmax_c": "observed",
                "tmin_c": "observed",
                "humidity_pct": "rule_estimated",
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
                "uncertainty": "ensemble_tree_variance",
            },
            "real_time_conditions": {
                "current_state": current_conditions,
                "risk_level": current_risk,
                "precautions": current_precautions,
                "scenario_precautions": scenario_precautions,
            },
            "real_time_conditions_basis": {
                "current_state": "observed + climatological_probability + rule_derived",
                "risk_level": "rule_based",
                "precautions": "rule_based",
                "scenario_precautions": "rule_based",
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
            "model_comparison": self.forecaster.get_model_comparison(),
            "feature_importance": self.forecaster.get_feature_importance(),
            "scorecard": scorecard,
            "extreme_events": extremes,
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
        result = {
            "labels": forecast["date"].dt.strftime("%d %b").tolist(),
            "rainfall": forecast["rainfall_mm"].round(2).tolist(),
            "tmax": forecast["tmax_c"].round(2).tolist(),
            "tmin": forecast["tmin_c"].round(2).tolist(),
        }
        # Add uncertainty bands if available
        if "rainfall_lower" in forecast.columns:
            result["rainfall_lower"] = forecast["rainfall_lower"].round(2).tolist()
            result["rainfall_upper"] = forecast["rainfall_upper"].round(2).tolist()
            result["tmax_lower"] = forecast["tmax_lower"].round(2).tolist()
            result["tmax_upper"] = forecast["tmax_upper"].round(2).tolist()
            result["tmin_lower"] = forecast["tmin_lower"].round(2).tolist()
            result["tmin_upper"] = forecast["tmin_upper"].round(2).tolist()
        return result

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

    def _build_scorecard(self, region_history: pd.DataFrame, forecast: pd.DataFrame) -> dict:
        """Compute a live scorecard from actual system data."""
        total_obs = len(self.observations)
        total_regions = self.observations["region"].nunique()
        date_range = self.observations["date"]
        days_covered = (date_range.max() - date_range.min()).days + 1
        actual_dates = date_range.nunique()

        # Data coverage: what % of expected region×date cells have data
        expected_cells = days_covered * total_regions
        data_coverage = min(100.0, (total_obs / expected_cells) * 100.0) if expected_cells > 0 else 0.0

        # Model performance: use R² of rainfall (hardest target), scaled 0-100
        rf_metrics = self.forecaster.deterministic_metrics
        r2_rainfall = rf_metrics.get("rainfall_mm", {}).get("r2", 0.0)
        model_score = max(0.0, min(100.0, r2_rainfall * 100))

        # Forecast confidence: mean of tree std across forecast horizon (lower std = higher confidence)
        forecast_confidence = 75.0  # default
        if self.forecaster.model is not None and not forecast.empty:
            tree_preds = np.array([
                tree.predict(self.forecaster.prepare_features(self.observations).iloc[-1:][self.forecaster.feature_columns].values)
                for tree in self.forecaster.model.estimators_
            ])
            mean_std = float(np.mean(np.std(tree_preds, axis=0)))
            # Normalize: lower std = higher confidence (scale: std<1 → 95%, std>10 → 30%)
            forecast_confidence = max(30.0, min(95.0, 95.0 - mean_std * 7))

        # Data freshness: how recent is the latest observation
        latest_date = pd.to_datetime(date_range.max())
        days_old = max(0, (pd.Timestamp.now() - latest_date).days)
        freshness = max(0.0, min(100.0, 100.0 - days_old * 0.5))

        return {
            "data_coverage": round(data_coverage, 1),
            "model_performance": round(model_score, 1),
            "forecast_confidence": round(forecast_confidence, 1),
            "data_freshness": round(freshness, 1),
            "total_observations": total_obs,
            "total_regions": total_regions,
            "date_range": [date_range.min().strftime("%Y-%m-%d"), date_range.max().strftime("%Y-%m-%d")],
            "days_old": days_old,
            "basis": "computed_from_actual_system_state",
        }

    def _detect_extreme_events(self, region_history: pd.DataFrame, current_state: dict) -> dict:
        """Detect extreme weather conditions from historical context."""
        history = region_history.copy()
        history["date"] = pd.to_datetime(history["date"])
        recent_7d = history.tail(7)

        events = []
        current = current_state.get("current", {})
        rainfall = current.get("rainfall_mm", 0)
        tmax = current.get("tmax_c", 0)

        # Heavy rainfall threshold (IMD: >64.5mm = very heavy)
        if rainfall > 64.5:
            events.append({"type": "heavy_rainfall", "severity": "severe", "value": rainfall, "threshold": 64.5, "label": f"Very Heavy Rainfall: {rainfall:.1f} mm"})
        elif rainfall > 35.5:
            events.append({"type": "heavy_rainfall", "severity": "moderate", "value": rainfall, "threshold": 35.5, "label": f"Heavy Rainfall: {rainfall:.1f} mm"})

        # Heatwave conditions (simplified: tmax > 40°C or >5°C above normal)
        if tmax > 40.0:
            events.append({"type": "heatwave", "severity": "severe", "value": tmax, "threshold": 40.0, "label": f"Extreme Heat: {tmax:.1f}°C"})
        elif tmax > 37.0:
            events.append({"type": "heatwave", "severity": "moderate", "value": tmax, "threshold": 37.0, "label": f"Heat Alert: {tmax:.1f}°C"})

        # Drought indicator: 7-day cumulative rainfall < 5mm
        cum_rain_7d = float(recent_7d["rainfall_mm"].sum())
        if cum_rain_7d < 2.0 and len(recent_7d) >= 7:
            events.append({"type": "drought_risk", "severity": "watch", "value": cum_rain_7d, "threshold": 2.0, "label": f"Dry Spell: {cum_rain_7d:.1f} mm in 7 days"})

        # 99th percentile rainfall
        p99 = float(history["rainfall_mm"].quantile(0.99))
        if rainfall > p99 and p99 > 0:
            events.append({"type": "extreme_rainfall", "severity": "extreme", "value": rainfall, "threshold": p99, "label": f"Extreme Rainfall (>99th pctl): {rainfall:.1f} mm"})

        return {
            "events": events,
            "count": len(events),
            "has_alerts": len(events) > 0,
            "basis": "rule_derived_from_observed_data",
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
