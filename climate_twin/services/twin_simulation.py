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


@dataclass(frozen=True)
class TwinReplayResult:
    """Store a historical replay used for validation and case-study reporting."""

    region: str
    start_date: str
    end_date: str
    series: pd.DataFrame
    summary: dict[str, float | str]
    basis: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "series": _replay_series_payload(self.series),
            "summary": self.summary,
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


def build_replay_result(
    forecaster: ClimateForecaster,
    region_history: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> TwinReplayResult:
    """Replay a historical date range with one-day-ahead predictions."""

    history = region_history.sort_values("date").reset_index(drop=True).copy()
    history["date"] = pd.to_datetime(history["date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for target_date in pd.date_range(start=start, end=end, freq="D"):
        prior_history = history.loc[history["date"] < target_date]
        actual = history.loc[history["date"] == target_date]
        if prior_history.empty or actual.empty:
            continue

        prediction = forecaster.predict_next(prior_history, horizon_days=1).iloc[0]
        rows.append(
            {
                "date": target_date,
                "actual_rainfall_mm": float(actual.iloc[0]["rainfall_mm"]),
                "predicted_rainfall_mm": float(prediction["rainfall_mm"]),
                "actual_tmax_c": float(actual.iloc[0]["tmax_c"]),
                "predicted_tmax_c": float(prediction["tmax_c"]),
                "actual_tmin_c": float(actual.iloc[0]["tmin_c"]),
                "predicted_tmin_c": float(prediction["tmin_c"]),
            }
        )

    series = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if series.empty:
        summary = {"records": 0, "mae_rainfall_mm": 0.0, "mae_tmax_c": 0.0, "mae_tmin_c": 0.0}
    else:
        summary = {
            "records": int(len(series)),
            "mae_rainfall_mm": float(np.abs(series["predicted_rainfall_mm"] - series["actual_rainfall_mm"]).mean()),
            "mae_tmax_c": float(np.abs(series["predicted_tmax_c"] - series["actual_tmax_c"]).mean()),
            "mae_tmin_c": float(np.abs(series["predicted_tmin_c"] - series["actual_tmin_c"]).mean()),
        }

    return TwinReplayResult(
        region=str(history.iloc[0]["region"]) if not history.empty else "unknown",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        series=series,
        summary=summary,
        basis={"series": "ml_forecast", "summary": "rule_derived"},
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


def _replay_series_payload(series: pd.DataFrame) -> dict[str, Any]:
    replay = series.copy()
    if replay.empty:
        return {"labels": [], "basis": "ml_forecast"}
    replay["date"] = pd.to_datetime(replay["date"])
    return {
        "labels": replay["date"].dt.strftime("%d %b %Y").tolist(),
        "actual_rainfall_mm": replay["actual_rainfall_mm"].round(2).tolist(),
        "predicted_rainfall_mm": replay["predicted_rainfall_mm"].round(2).tolist(),
        "actual_tmax_c": replay["actual_tmax_c"].round(2).tolist(),
        "predicted_tmax_c": replay["predicted_tmax_c"].round(2).tolist(),
        "actual_tmin_c": replay["actual_tmin_c"].round(2).tolist(),
        "predicted_tmin_c": replay["predicted_tmin_c"].round(2).tolist(),
        "basis": "ml_forecast",
    }
