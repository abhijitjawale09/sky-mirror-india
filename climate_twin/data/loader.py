from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

DOWNLOAD_DIR = Path.home() / "Downloads"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "climate_training_data.csv"

SOURCE_GROUPS = {
    "rainfall": [DOWNLOAD_DIR / "rainfall_2024_long.csv", DOWNLOAD_DIR / "rainfall_2025_long.csv"],
    "tmax": [DOWNLOAD_DIR / "maxtemp_2024_long.csv", DOWNLOAD_DIR / "maxtemp_2025_long.csv"],
    "tmin": [DOWNLOAD_DIR / "mintemp_2024_long.csv", DOWNLOAD_DIR / "mintemp_2025_long.csv"],
}


def get_region_profile(region_name: str) -> RegionProfile:
    for profile in REGION_PROFILES:
        if profile.name == region_name:
            return profile
    return REGION_PROFILES[0]


def _nearest_region(latitude: float, longitude: float) -> str:
    return min(
        REGION_PROFILES,
        key=lambda profile: (profile.latitude - latitude) ** 2 + (profile.longitude - longitude) ** 2,
    ).name


def _load_collection(paths: list[Path], value_column: str, rename_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.rename(columns=rename_map)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frames.append(frame[["date", "latitude", "longitude", value_column]])

    if not frames:
        missing = ", ".join(str(path.name) for path in paths)
        raise FileNotFoundError(f"Could not find any source files for {missing} in {DOWNLOAD_DIR}")

    return pd.concat(frames, ignore_index=True)


def build_fused_dataset() -> pd.DataFrame:
    rainfall = _load_collection(SOURCE_GROUPS["rainfall"], "rainfall_mm", {"lat": "latitude", "lon": "longitude"})
    tmax = _load_collection(SOURCE_GROUPS["tmax"], "tmax_c", {"lat": "latitude", "lon": "longitude", "max_temp_c": "tmax_c"})
    tmin = _load_collection(SOURCE_GROUPS["tmin"], "tmin_c", {"lat": "latitude", "lon": "longitude", "min_temp_c": "tmin_c"})

    fused = rainfall.merge(tmax, on=["date", "latitude", "longitude"], how="inner").merge(
        tmin, on=["date", "latitude", "longitude"], how="inner"
    )
    fused = fused.dropna().sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)

    fused["region"] = fused.apply(lambda row: _nearest_region(float(row["latitude"]), float(row["longitude"])), axis=1)
    fused["diurnal_range_c"] = fused["tmax_c"] - fused["tmin_c"]
    fused["humidity_pct"] = np.clip(92.0 - fused["diurnal_range_c"] * 2.15 + np.log1p(fused["rainfall_mm"]) * 3.0, 35.0, 98.0)
    fused["month"] = fused["date"].dt.month
    fused["day_of_year"] = fused["date"].dt.dayofyear
    fused["month_sin"] = np.sin(2 * np.pi * fused["month"] / 12.0)
    fused["month_cos"] = np.cos(2 * np.pi * fused["month"] / 12.0)

    region_daily = (
        fused.groupby(["date", "region"], as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            tmax_c=("tmax_c", "mean"),
            tmin_c=("tmin_c", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            diurnal_range_c=("diurnal_range_c", "mean"),
            grid_points=("rainfall_mm", "size"),
        )
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    region_daily["day_of_year"] = region_daily["date"].dt.dayofyear
    region_daily["month"] = region_daily["date"].dt.month
    region_daily["month_sin"] = np.sin(2 * np.pi * region_daily["month"] / 12.0)
    region_daily["month_cos"] = np.cos(2 * np.pi * region_daily["month"] / 12.0)
    region_daily["rain_intensity"] = np.log1p(region_daily["rainfall_mm"])
    region_daily["temperature_gap"] = region_daily["tmax_c"] - region_daily["tmin_c"]

    region_daily["region_code"] = region_daily["region"].astype("category").cat.codes
    region_daily = region_daily.drop(columns=["month"])
    return region_daily


def save_processed_dataset(frame: pd.DataFrame, path: Path | None = None) -> Path:
    output_path = path or PROCESSED_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_training_observations() -> tuple[pd.DataFrame, str]:
    if PROCESSED_PATH.exists():
        frame = pd.read_csv(PROCESSED_PATH)
        normalized = normalize_observations(frame)
        return normalized, f"processed:{PROCESSED_PATH.relative_to(PROJECT_ROOT)}"

    fused = build_fused_dataset()
    processed_path = save_processed_dataset(fused)
    normalized = normalize_observations(fused)
    return normalized, f"generated:{processed_path.relative_to(PROJECT_ROOT)}"


def latest_snapshot(observations: pd.DataFrame, region_name: str) -> dict[str, float | str]:
    region_df = observations.loc[observations["region"] == region_name].sort_values("date")
    latest = region_df.iloc[-1]
    previous = region_df.iloc[-8:-1]

    rainfall_baseline = previous["rainfall_mm"].mean() if not previous.empty else latest["rainfall_mm"]
    tmax_baseline = previous["tmax_c"].mean() if not previous.empty else latest["tmax_c"]

    return {
        "region": region_name,
        "date": pd.to_datetime(latest["date"]).strftime("%d %b %Y"),
        "rainfall_mm": float(latest["rainfall_mm"]),
        "tmax_c": float(latest["tmax_c"]),
        "tmin_c": float(latest["tmin_c"]),
        "humidity_pct": float(latest["humidity_pct"]),
        "rainfall_anomaly": float(latest["rainfall_mm"] - rainfall_baseline),
        "temperature_anomaly": float(latest["tmax_c"] - tmax_baseline),
        "latitude": float(latest["latitude"]),
        "longitude": float(latest["longitude"]),
    }


def normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "date",
        "region",
        "latitude",
        "longitude",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "humidity_pct",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {', '.join(missing)}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["region"] = normalized["region"].astype(str)
    numeric_columns = ["latitude", "longitude", "rainfall_mm", "tmax_c", "tmin_c", "humidity_pct"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "grid_points" not in normalized.columns:
        normalized["grid_points"] = 1
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
    if "region_code" not in normalized.columns:
        normalized["region_code"] = normalized["region"].astype("category").cat.codes

    normalized = normalized.dropna(subset=list(required_columns)).sort_values(["region", "date"]).reset_index(drop=True)
    return normalized
