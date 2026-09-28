from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .forecasting import ClimateForecaster
from .twin_state import compare_risk_levels, daily_same_day_normal, risk_level_from_rainfall_series


@dataclass(frozen=True)
class TwinScenario:
    """Define a what-if climate perturbation applied on top of the ML forecast."""

    rainfall_delta_pct: float = 0.0
    temp_delta_c: float = 0.0
    horizon_days: int = 14


@dataclass(frozen=True)
class TwinSimulationResult:
    """Store the baseline forecast, scenario forecast, and rule-derived comparisons."""

    scenario: TwinScenario
    baseline_forecast: pd.DataFrame
    scenario_forecast: pd.DataFrame
    cumulative_normal_rainfall: float
    cumulative_baseline_rainfall: float
    cumulative_scenario_rainfall: float
    rainfall_deficit_surplus: float
    baseline_risk_level: str
    scenario_risk_level: str
    risk_comparison: dict[str, str | int]
    basis: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": {
                "rainfall_delta_pct": self.scenario.rainfall_delta_pct,
                "temp_delta_c": self.scenario.temp_delta_c,
                "horizon_days": self.scenario.horizon_days,
            },
            "baseline_forecast": _forecast_payload(self.baseline_forecast, basis="ml_forecast"),
            "scenario_forecast": _forecast_payload(self.scenario_forecast, basis="ml_forecast"),
            "cumulative_normal_rainfall": self.cumulative_normal_rainfall,
            "cumulative_baseline_rainfall": self.cumulative_baseline_rainfall,
            "cumulative_scenario_rainfall": self.cumulative_scenario_rainfall,
            "rainfall_deficit_surplus": self.rainfall_deficit_surplus,
            "baseline_risk_level": self.baseline_risk_level,
            "scenario_risk_level": self.scenario_risk_level,
            "risk_comparison": self.risk_comparison,
            "basis": self.basis,
        }


def build_simulation_result(
    forecaster: ClimateForecaster,
    region_history: pd.DataFrame,
    scenario: TwinScenario,
) -> TwinSimulationResult:
    """Run the ML forecast and apply rule-based scenario post-processing."""

    baseline_forecast = forecaster.predict_next(region_history, horizon_days=scenario.horizon_days)
    scenario_forecast = forecaster.predict_next(
        region_history,
        horizon_days=scenario.horizon_days,
        rainfall_delta_pct=scenario.rainfall_delta_pct,
        temp_delta_c=scenario.temp_delta_c,
    )

    normal_total = 0.0
    baseline_total = 0.0
    scenario_total = 0.0

    baseline_risk_level = "normal"
    scenario_risk_level = "normal"

    baseline_levels: list[str] = []
    scenario_levels: list[str] = []

    for row in baseline_forecast.itertuples(index=False):
        normals = daily_same_day_normal(region_history, pd.to_datetime(row.date))
        normal_total += normals["rainfall_mm"]
        baseline_total += float(row.rainfall_mm)
        baseline_levels.append(risk_level_from_rainfall_series(region_history, pd.to_datetime(row.date), float(row.rainfall_mm)))

    for row in scenario_forecast.itertuples(index=False):
        scenario_total += float(row.rainfall_mm)
        scenario_levels.append(risk_level_from_rainfall_series(region_history, pd.to_datetime(row.date), float(row.rainfall_mm)))

    if baseline_levels:
        baseline_risk_level = baseline_levels[-1]
    if scenario_levels:
        scenario_risk_level = scenario_levels[-1]

    rainfall_deficit_surplus = float(scenario_total - normal_total)
    return TwinSimulationResult(
        scenario=scenario,
        baseline_forecast=baseline_forecast,
        scenario_forecast=scenario_forecast,
        cumulative_normal_rainfall=float(normal_total),
        cumulative_baseline_rainfall=float(baseline_total),
        cumulative_scenario_rainfall=float(scenario_total),
        rainfall_deficit_surplus=rainfall_deficit_surplus,
        baseline_risk_level=baseline_risk_level,
        scenario_risk_level=scenario_risk_level,
        risk_comparison=compare_risk_levels(baseline_risk_level, scenario_risk_level),
        basis={
            "baseline_forecast": "ml_forecast",
            "scenario_forecast": "ml_forecast",
            "cumulative_normal_rainfall": "rule_derived",
            "cumulative_baseline_rainfall": "ml_forecast",
            "cumulative_scenario_rainfall": "ml_forecast",
            "rainfall_deficit_surplus": "rule_derived",
            "baseline_risk_level": "rule_derived",
            "scenario_risk_level": "rule_derived",
            "risk_comparison": "rule_derived",
        },
    )


def _forecast_payload(forecast: pd.DataFrame, basis: str) -> dict[str, Any]:
    forecast_frame = forecast.copy()
    forecast_frame["date"] = pd.to_datetime(forecast_frame["date"])
    return {
        "labels": forecast_frame["date"].dt.strftime("%d %b").tolist(),
        "rainfall_mm": forecast_frame["rainfall_mm"].round(2).tolist(),
        "tmax_c": forecast_frame["tmax_c"].round(2).tolist(),
        "tmin_c": forecast_frame["tmin_c"].round(2).tolist(),
        "basis": basis,
    }
