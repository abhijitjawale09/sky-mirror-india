from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealTimeConditionResult:
    """Container for the JSON-serializable state returned by the real-time conditions layer.

    This module intentionally treats the latest date in the loaded historical table as
    'today' for the dashboard. That is a climatological presentation layer, not a real
    live meteorological feed. The data is not streaming from an external satellite or
    station feed; it is a derived state computed from the historical daily dataset.
    """

    region: str
    date: str
    current: dict[str, float]
    rainfall_probability_pct: float
    anomaly: dict[str, float]
    risk_level: str
    reason: str
    precautions: list[str]
    basis: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "date": self.date,
            "current": self.current,
            "rainfall_probability_pct": self.rainfall_probability_pct,
            "anomaly": self.anomaly,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "precautions": self.precautions,
            "basis": self.basis,
        }


class RealTimeConditions:
    """Compute a current climate snapshot from the historical dataset.

    The method intentionally uses the latest date present in the historical zone as the
    reference 'today'. This is not a real-time telemetry system; it is an operational
    digital-twin summary that interprets the most recent available daily observations in
    context of the 15-year climatology. The probability estimate is explicitly labeled
    as climatological_probability because it is derived from historical same-day-of-year
    frequencies rather than a physical weather forecast.
    """

    def __init__(self, observations: pd.DataFrame) -> None:
        self.observations = observations.copy()
        self.observations["date"] = pd.to_datetime(self.observations["date"])
        self.observations = self.observations.sort_values(["region", "date"]).reset_index(drop=True)

    def get_current_state(self, region: str) -> dict[str, Any]:
        """Return current observed conditions and a climatological probability estimate.

        For the digital twin dashboard, the latest available row for the region is treated
        as the current state. The rainfall probability is computed from the same day-of-year
        climatology within +/- 7 days across all 15 years, which is a historical estimate
        of how often similar late-December / pre-monsoon dates had rainfall above a threshold.
        This is not a forecast model output and is purposely labeled as such.
        """

        region_history = self._region_history(region)
        if region_history.empty:
            raise ValueError(f"No observations found for region '{region}'")

        latest = region_history.iloc[-1]
        current_date = pd.to_datetime(latest["date"])
        day_of_year = int(current_date.dayofyear)
        same_day_window = self.observations.loc[
            (self.observations["region"] == region)
            & (self.observations["date"].dt.dayofyear.between(day_of_year - 7, day_of_year + 7, inclusive="both"))
        ].copy()

        if same_day_window.empty:
            same_day_window = region_history.copy()

        rainfall_threshold = 2.5
        probability = float(
            np.mean(same_day_window["rainfall_mm"].to_numpy() > rainfall_threshold) * 100.0
        ) if not same_day_window.empty else 0.0

        anomaly = self._anomaly_for_date(region_history, current_date)
        current = {
            "rainfall_mm": float(latest["rainfall_mm"]),
            "tmax_c": float(latest["tmax_c"]),
            "tmin_c": float(latest["tmin_c"]),
            "humidity_pct": float(latest.get("humidity_pct", 0.0)),
        }

        return {
            "region": region,
            "date": current_date.strftime("%Y-%m-%d"),
            "current": current,
            "rainfall_probability_pct": round(probability, 2),
            "anomaly": {
                "rainfall_mm_pct": round(anomaly["rainfall_mm_pct"], 2),
                "tmax_c_delta_c": round(anomaly["tmax_c_delta_c"], 2),
                "tmin_c_delta_c": round(anomaly["tmin_c_delta_c"], 2),
            },
            "basis": {
                "current": "observed",
                "rainfall_probability_pct": "climatological_probability",
                "anomaly": "rule_derived",
            },
        }

    def get_risk_level(self, current_state: dict[str, Any]) -> dict[str, Any]:
        """Classify the current conditions into a simple risk level.

        The thresholds are intentionally simple percentile-based rules, not a learned model.
        They use the historical behavior for the same day-of-year profile to rank the latest
        observed climate values relative to the region's own climatology over 15 years.
        """

        region = str(current_state["region"])
        region_history = self._region_history(region)
        if region_history.empty:
            return {"region": region, "risk_level": "normal", "reason": "No historical data available", "basis": "rule_based"}

        today = pd.to_datetime(current_state["date"])
        day_of_year = int(today.dayofyear)
        window = region_history.loc[
            region_history["date"].dt.dayofyear.between(day_of_year - 7, day_of_year + 7, inclusive="both")
        ].copy()
        if window.empty:
            window = region_history

        current = current_state["current"]
        rainfall_value = float(current["rainfall_mm"])
        tmax_value = float(current["tmax_c"])
        tmin_value = float(current["tmin_c"])

        rainfall_p = self._percentile(window["rainfall_mm"], rainfall_value)
        tmax_p = self._percentile(window["tmax_c"], tmax_value)
        tmin_p = self._percentile(window["tmin_c"], tmin_value)

        if rainfall_p >= 0.9 or tmax_p >= 0.9:
            level = "alert"
            reason = self._build_alert_reason(rainfall_value, tmax_value, window, "high_risk")
        elif rainfall_p >= 0.8 or tmax_p >= 0.8:
            level = "watch"
            reason = self._build_alert_reason(rainfall_value, tmax_value, window, "watch")
        elif rainfall_p <= 0.1 or tmax_p <= 0.1:
            level = "alert"
            reason = self._build_alert_reason(rainfall_value, tmax_value, window, "drought_risk")
        else:
            level = "normal"
            reason = "Within the 15-year same-date climatological range for this region"

        return {
            "region": region,
            "risk_level": level,
            "reason": reason,
            "basis": "rule_based",
        }

    def get_precautions(self, risk_level: str, scenario_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a list of precautionary messages from a documented rule table.

        The lookup table is intentionally editable and separate from the model. This keeps the
        logic transparent and suitable for reporting: it is a rule-based advisory layer, not an
        artificial intelligence decision engine.
        """

        scenario_context = scenario_context or {}
        rules = {
            "normal": [
                "Continue routine monitoring and seasonal planning",
                "Maintain standard water and crop checks",
            ],
            "watch": [
                "Increase field and infrastructure monitoring",
                "Keep drainage channels clear and inspect vulnerable spots",
            ],
            "alert": [
                "Activate emergency preparedness checks",
                "Prioritize public safety and local advisories",
            ],
            "flood": [
                "Avoid low-lying areas",
                "Check drainage and flood barriers",
                "Monitor local flood advisories",
            ],
            "heat": [
                "Stay hydrated during peak hours",
                "Avoid outdoor work from 12-3pm",
                "Watch for heat stress in crops and livestock",
            ],
            "drought": [
                "Prioritize water storage",
                "Delay water-intensive irrigation",
                "Monitor reservoir levels and moisture stress",
            ],
        }

        messages: list[str] = []
        lower_level = str(risk_level).lower()

        if lower_level in {"alert", "watch"}:
            if float(scenario_context.get("rainfall_delta_pct", 0.0)) > 10 or float(scenario_context.get("predicted_rainfall", 0.0)) > 70:
                messages.extend(rules["flood"])
            if float(scenario_context.get("temp_delta_c", 0.0)) > 1.0 or float(scenario_context.get("predicted_tmax", 0.0)) > 34:
                messages.extend(rules["heat"])
            if float(scenario_context.get("rainfall_delta_pct", 0.0)) < -10 or float(scenario_context.get("predicted_rainfall", 0.0)) < 10:
                messages.extend(rules["drought"])

        if not messages:
            messages = rules.get(lower_level, rules["normal"]).copy()

        return {
            "risk_level": risk_level,
            "messages": messages,
            "basis": "rule_based",
        }

    def _region_history(self, region: str) -> pd.DataFrame:
        return self.observations.loc[self.observations["region"].astype(str) == str(region)].sort_values("date").reset_index(drop=True)

    def _anomaly_for_date(self, region_history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
        day_of_year = int(target_date.dayofyear)
        same_day = region_history.loc[region_history["date"].dt.dayofyear == day_of_year].copy()
        if same_day.empty:
            same_day = region_history.copy()

        latest = region_history.iloc[-1]
        rainfall_avg = float(same_day["rainfall_mm"].mean())
        tmax_avg = float(same_day["tmax_c"].mean())
        tmin_avg = float(same_day["tmin_c"].mean())

        rainfall_pct = 0.0 if rainfall_avg == 0 else ((float(latest["rainfall_mm"]) - rainfall_avg) / rainfall_avg) * 100.0
        tmax_delta = float(latest["tmax_c"]) - tmax_avg
        tmin_delta = float(latest["tmin_c"]) - tmin_avg

        return {
            "rainfall_mm_pct": rainfall_pct,
            "tmax_c_delta_c": tmax_delta,
            "tmin_c_delta_c": tmin_delta,
        }

    def _percentile(self, series: pd.Series, value: float) -> float:
        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if cleaned.empty:
            return 0.5
        return float((cleaned <= value).mean())

    def _build_alert_reason(self, rainfall_value: float, tmax_value: float, window: pd.DataFrame, risk_type: str) -> str:
        rainfall_avg = float(window["rainfall_mm"].mean())
        tmax_avg = float(window["tmax_c"].mean())
        if risk_type in {"high_risk", "watch"}:
            return (
                f"rainfall {rainfall_value:.1f} mm and tmax {tmax_value:.1f}°C are elevated relative to the 15-year "
                f"same-date mean (rainfall {rainfall_avg:.1f} mm, tmax {tmax_avg:.1f}°C)"
            )
        if risk_type == "drought_risk":
            return (
                f"rainfall {rainfall_value:.1f} mm is well below the 15-year same-date mean "
                f"({rainfall_avg:.1f} mm) and heat is elevated relative to the historical range"
            )
        return "Historical risk threshold breached for this date"
