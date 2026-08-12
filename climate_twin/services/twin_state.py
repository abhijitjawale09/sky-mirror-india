from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RISK_LEVEL_ORDER = {"normal": 0, "watch": 1, "alert": 2}


@dataclass(frozen=True)
class TwinState:
    """Represent the current climate state for a region.

    The state combines the latest observation, short and medium rolling windows,
    same-day-of-year climatology, anomalies, and a simple rainfall-driven risk
    interpretation. This is the live anchor for the digital twin.
    """

    region: str
    date: str
    latest_observed: dict[str, float]
    rolling_7d: dict[str, float]
    rolling_30d: dict[str, float]
    baseline_same_day: dict[str, float]
    anomaly_vs_baseline: dict[str, float]
    percentile_rank: dict[str, float]
    risk_flag: str
    risk_level: str
    basis: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "date": self.date,
            "latest_observed": self.latest_observed,
            "rolling_7d": self.rolling_7d,
            "rolling_30d": self.rolling_30d,
            "baseline_same_day": self.baseline_same_day,
            "anomaly_vs_baseline": self.anomaly_vs_baseline,
            "percentile_rank": self.percentile_rank,
            "risk_flag": self.risk_flag,
            "risk_level": self.risk_level,
            "basis": self.basis,
        }


def build_twin_state(region_history: pd.DataFrame) -> TwinState:
    """Build the live state from the most recent region history."""

    history = region_history.sort_values("date").reset_index(drop=True).copy()
    history["date"] = pd.to_datetime(history["date"])
    latest = history.iloc[-1]
    current_date = pd.to_datetime(latest["date"])
    current_day_of_year = int(current_date.dayofyear)

    recent_7 = history.tail(7)
    recent_30 = history.tail(30)
    same_day_history = history.loc[(history["date"] < current_date) & (history["date"].dt.dayofyear == current_day_of_year)]
    if same_day_history.empty:
        same_day_history = history.loc[history["date"] < current_date]

    latest_observed = {
        "rainfall_mm": float(latest["rainfall_mm"]),
        "tmax_c": float(latest["tmax_c"]),
        "tmin_c": float(latest["tmin_c"]),
        "humidity_pct": float(latest["humidity_pct"]),
    }
    rolling_7d = {
        "rainfall_mm": float(recent_7["rainfall_mm"].mean()),
        "tmax_c": float(recent_7["tmax_c"].mean()),
        "tmin_c": float(recent_7["tmin_c"].mean()),
        "humidity_pct": float(recent_7["humidity_pct"].mean()),
    }
    rolling_30d = {
        "rainfall_mm": float(recent_30["rainfall_mm"].mean()),
        "tmax_c": float(recent_30["tmax_c"].mean()),
        "tmin_c": float(recent_30["tmin_c"].mean()),
        "humidity_pct": float(recent_30["humidity_pct"].mean()),
    }
    baseline_same_day = {
        "rainfall_mm": _safe_mean(same_day_history["rainfall_mm"], latest_observed["rainfall_mm"]),
        "tmax_c": _safe_mean(same_day_history["tmax_c"], latest_observed["tmax_c"]),
        "tmin_c": _safe_mean(same_day_history["tmin_c"], latest_observed["tmin_c"]),
        "humidity_pct": _safe_mean(same_day_history["humidity_pct"], latest_observed["humidity_pct"]),
    }
    anomaly_vs_baseline = {
        key: float(latest_observed[key] - baseline_same_day[key]) for key in baseline_same_day
    }
    percentile_rank = {
        "rainfall_mm": _percentile_rank(same_day_history["rainfall_mm"], latest_observed["rainfall_mm"]),
        "tmax_c": _percentile_rank(same_day_history["tmax_c"], latest_observed["tmax_c"]),
        "tmin_c": _percentile_rank(same_day_history["tmin_c"], latest_observed["tmin_c"]),
    }
    risk_flag, risk_level = _derive_risk_from_percentile(percentile_rank["rainfall_mm"])

    return TwinState(
        region=str(latest["region"]),
        date=current_date.strftime("%Y-%m-%d"),
        latest_observed=latest_observed,
        rolling_7d=rolling_7d,
        rolling_30d=rolling_30d,
        baseline_same_day=baseline_same_day,
        anomaly_vs_baseline=anomaly_vs_baseline,
        percentile_rank=percentile_rank,
        risk_flag=risk_flag,
        risk_level=risk_level,
        basis={
            "latest_observed": "observed",
            "rolling_7d": "observed",
            "rolling_30d": "observed",
            "baseline_same_day": "rule_derived",
            "anomaly_vs_baseline": "rule_derived",
            "percentile_rank": "rule_derived",
            "risk_flag": "rule_derived",
            "risk_level": "rule_derived",
        },
    )


def daily_same_day_normal(region_history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
    """Return same-day-of-year normals for a target date."""

    history = region_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    day_of_year = int(pd.to_datetime(target_date).dayofyear)
    same_day = history.loc[history["date"].dt.dayofyear == day_of_year]
    if same_day.empty:
        same_day = history
    return {
        "rainfall_mm": _safe_mean(same_day["rainfall_mm"], 0.0),
        "tmax_c": _safe_mean(same_day["tmax_c"], 0.0),
        "tmin_c": _safe_mean(same_day["tmin_c"], 0.0),
        "humidity_pct": _safe_mean(same_day["humidity_pct"], 0.0),
    }


def risk_level_from_rainfall_series(region_history: pd.DataFrame, target_date: pd.Timestamp, rainfall_value: float) -> str:
    """Map a rainfall forecast to a discrete risk level using same-day history."""

    history = region_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    day_of_year = int(pd.to_datetime(target_date).dayofyear)
    same_day = history.loc[history["date"].dt.dayofyear == day_of_year]["rainfall_mm"]
    percentile = _percentile_rank(same_day, rainfall_value)
    _, risk_level = _derive_risk_from_percentile(percentile)
    return risk_level


def compare_risk_levels(baseline_level: str, scenario_level: str) -> dict[str, str | int]:
    """Compare baseline and scenario risk levels for the what-if layer."""

    baseline_rank = RISK_LEVEL_ORDER.get(baseline_level, 0)
    scenario_rank = RISK_LEVEL_ORDER.get(scenario_level, 0)
    if scenario_rank > baseline_rank:
        trend = "higher"
    elif scenario_rank < baseline_rank:
        trend = "lower"
    else:
        trend = "unchanged"
    return {
        "baseline_level": baseline_level,
        "scenario_level": scenario_level,
        "baseline_rank": baseline_rank,
        "scenario_rank": scenario_rank,
        "trend": trend,
    }


def _safe_mean(series: pd.Series, fallback: float) -> float:
    value = float(series.mean()) if not series.empty else float(fallback)
    if np.isnan(value):
        return float(fallback)
    return value


def _percentile_rank(series: pd.Series, value: float) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return 0.5
    return float((cleaned <= value).mean())


def _derive_risk_from_percentile(percentile: float) -> tuple[str, str]:
    if percentile <= 0.1:
        return "drought_risk", "alert"
    if percentile <= 0.2:
        return "drought_watch", "watch"
    if percentile >= 0.9:
        return "flood_risk", "alert"
    if percentile >= 0.8:
        return "flood_watch", "watch"
    return "stable", "normal"
