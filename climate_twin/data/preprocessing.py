from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegionProfile:
    name: str
    latitude: float
    longitude: float


REGION_PROFILES = [
    RegionProfile("Kerala Coast", 9.5, 76.5),
    RegionProfile("Indo-Gangetic Plain", 26.5, 81.0),
    RegionProfile("Northeast", 26.0, 91.7),
    RegionProfile("Central India", 22.0, 79.0),
    RegionProfile("Deccan Plateau", 17.8, 78.5),
    RegionProfile("Rajasthan Desert", 27.0, 73.0),
    RegionProfile("Coastal Odisha", 20.5, 85.5),
]

REQUIRED_RAW_COLUMNS = {
    "date",
    "region",
    "latitude",
    "longitude",
    "rainfall_mm",
    "tmax_c",
    "tmin_c",
    "humidity_pct",
}

FEATURE_COLUMNS = [
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
    "insat_lst_lag_1",
    "insat_sst_lag_1",
    "latitude",
    "longitude",
    "region_code",
]

TARGET_COLUMNS = ["rainfall_mm", "tmax_c", "tmin_c"]


def get_region_profile(region_name: str) -> RegionProfile:
    for profile in REGION_PROFILES:
        if profile.name == region_name:
            return profile
    return REGION_PROFILES[0]


def build_region_code_map(regions: list[str]) -> dict[str, int]:
    ordered_regions = sorted(set(regions))
    return {region: index for index, region in enumerate(ordered_regions)}


def normalize_raw_observations(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["region"] = normalized["region"].astype(str)

    numeric_columns = ["latitude", "longitude", "rainfall_mm", "tmax_c", "tmin_c", "humidity_pct"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "diurnal_range_c" not in normalized.columns:
        normalized["diurnal_range_c"] = normalized["tmax_c"] - normalized["tmin_c"]
    if "day_of_year" not in normalized.columns:
        normalized["day_of_year"] = normalized["date"].dt.dayofyear
    if "month_sin" not in normalized.columns or "month_cos" not in normalized.columns:
        month = normalized["date"].dt.month
        normalized["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        normalized["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    if "rain_intensity" not in normalized.columns:
        normalized["rain_intensity"] = np.log1p(normalized["rainfall_mm"].clip(lower=0))
    if "temperature_gap" not in normalized.columns:
        normalized["temperature_gap"] = normalized["tmax_c"] - normalized["tmin_c"]
    if "grid_points" not in normalized.columns:
        normalized["grid_points"] = 1

    # INSAT columns: fill with NaN if absent (regions without INSAT coverage)
    for insat_col in ["insat_lst_mean_K", "insat_sst_mean_K"]:
        if insat_col not in normalized.columns:
            normalized[insat_col] = np.nan
        else:
            normalized[insat_col] = pd.to_numeric(normalized[insat_col], errors="coerce")

    normalized = normalized.dropna(subset=list(REQUIRED_RAW_COLUMNS))
    normalized = normalized.sort_values(["region", "date"]).reset_index(drop=True)
    return normalized


def add_lag_features(frame: pd.DataFrame, region_code_map: dict[str, int] | None = None) -> pd.DataFrame:
    frame = frame.sort_values(["region", "date"]).copy()

    if region_code_map is None:
        region_code_map = build_region_code_map(frame["region"].astype(str).unique().tolist())
    frame["region_code"] = frame["region"].map(region_code_map).fillna(0).astype(int)

    lag_groups = frame.groupby("region", group_keys=False)

    frame["rainfall_lag_1"] = lag_groups["rainfall_mm"].shift(1)
    frame["rainfall_lag_7"] = lag_groups["rainfall_mm"].shift(7)
    frame["rainfall_roll_7"] = lag_groups["rainfall_mm"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=3).mean()
    )
    frame["tmax_lag_1"] = lag_groups["tmax_c"].shift(1)
    frame["tmax_roll_7"] = lag_groups["tmax_c"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=3).mean()
    )
    frame["tmin_lag_1"] = lag_groups["tmin_c"].shift(1)
    frame["humidity_lag_1"] = lag_groups["humidity_pct"].shift(1)
    frame["diurnal_range_lag_1"] = lag_groups["diurnal_range_c"].shift(1)

    # INSAT lag features (NaN for regions without satellite coverage)
    if "insat_lst_mean_K" in frame.columns:
        frame["insat_lst_lag_1"] = lag_groups["insat_lst_mean_K"].shift(1)
    else:
        frame["insat_lst_lag_1"] = np.nan
    if "insat_sst_mean_K" in frame.columns:
        frame["insat_sst_lag_1"] = lag_groups["insat_sst_mean_K"].shift(1)
    else:
        frame["insat_sst_lag_1"] = np.nan

    return frame


def prepare_training_features(
    observations: pd.DataFrame,
    region_code_map: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized = normalize_raw_observations(observations)

    if region_code_map is None:
        region_code_map = build_region_code_map(normalized["region"].astype(str).unique().tolist())

    with_lags = add_lag_features(normalized, region_code_map)
    # Drop rows where core lag features are NaN but keep INSAT NaN (imputed to 0 for RF)
    core_lag_cols = [c for c in FEATURE_COLUMNS if c not in ("insat_lst_lag_1", "insat_sst_lag_1")]
    with_lags = with_lags.dropna(subset=core_lag_cols).reset_index(drop=True)
    # Fill INSAT NaN with 0 so the Random Forest can handle missing satellite coverage
    for c in ["insat_lst_lag_1", "insat_sst_lag_1"]:
        with_lags[c] = with_lags[c].fillna(0.0)

    feature_frame = with_lags[FEATURE_COLUMNS + TARGET_COLUMNS].copy()
    return feature_frame, region_code_map


def prepare_inference_features(
    history: pd.DataFrame,
    region_code_map: dict[str, int],
    target_date: pd.Timestamp,
    step_ahead: int = 1,
) -> pd.DataFrame:
    sorted_history = history.sort_values("date").reset_index(drop=True)
    last_row = sorted_history.iloc[-1]
    region_code = region_code_map.get(str(last_row["region"]), 0)

    forecast_date = pd.to_datetime(last_row["date"]) + pd.Timedelta(days=step_ahead)
    month = forecast_date.month
    tail = sorted_history.tail(7)
    diurnal_tail = tail["diurnal_range_c"] if "diurnal_range_c" in tail.columns else tail["tmax_c"] - tail["tmin_c"]

    if step_ahead == 0:
        insat_lst_val = last_row.get("insat_lst_lag_1", last_row.get("insat_lst_mean_K", 0.0))
        insat_sst_val = last_row.get("insat_sst_lag_1", last_row.get("insat_sst_mean_K", 0.0))
    else:
        insat_lst_val = last_row.get("insat_lst_mean_K", last_row.get("insat_lst_lag_1", 0.0))
        insat_sst_val = last_row.get("insat_sst_mean_K", last_row.get("insat_sst_lag_1", 0.0))

    feature_row = {
        "day_of_year": forecast_date.dayofyear,
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "rainfall_lag_1": float(last_row["rainfall_mm"]) if step_ahead > 0 else float(last_row.get("rainfall_lag_1", last_row["rainfall_mm"])),
        "rainfall_lag_7": float(sorted_history.iloc[-7]["rainfall_mm"]) if len(sorted_history) >= 7 and step_ahead > 0 else float(last_row.get("rainfall_lag_7", last_row["rainfall_mm"])),
        "rainfall_roll_7": float(tail["rainfall_mm"].mean()) if step_ahead > 0 else float(last_row.get("rainfall_roll_7", tail["rainfall_mm"].mean())),
        "tmax_lag_1": float(last_row["tmax_c"]) if step_ahead > 0 else float(last_row.get("tmax_lag_1", last_row["tmax_c"])),
        "tmax_roll_7": float(tail["tmax_c"].mean()) if step_ahead > 0 else float(last_row.get("tmax_roll_7", tail["tmax_c"].mean())),
        "tmin_lag_1": float(last_row["tmin_c"]) if step_ahead > 0 else float(last_row.get("tmin_lag_1", last_row["tmin_c"])),
        "humidity_lag_1": float(last_row["humidity_pct"]) if step_ahead > 0 else float(last_row.get("humidity_lag_1", last_row["humidity_pct"])),
        "diurnal_range_lag_1": float(diurnal_tail.iloc[-1] if hasattr(diurnal_tail, "iloc") else diurnal_tail) if step_ahead > 0 else float(last_row.get("diurnal_range_lag_1", last_row["tmax_c"] - last_row["tmin_c"])),
        "insat_lst_lag_1": float(insat_lst_val) if pd.notna(insat_lst_val) else 0.0,
        "insat_sst_lag_1": float(insat_sst_val) if pd.notna(insat_sst_val) else 0.0,
        "latitude": float(last_row["latitude"]),
        "longitude": float(last_row["longitude"]),
        "region_code": int(region_code),
    }

    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def load_and_prepare_training_data(
    csv_path: Path | str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(csv_path)
    return prepare_training_features(frame)