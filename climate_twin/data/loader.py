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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "climate_training_data.csv"


def get_region_profile(region_name: str) -> RegionProfile:
    for profile in REGION_PROFILES:
        if profile.name == region_name:
            return profile
    return REGION_PROFILES[0]


def save_processed_dataset(frame: pd.DataFrame, path: Path | None = None) -> Path:
    output_path = path or PROCESSED_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_training_observations() -> tuple[pd.DataFrame, str]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed IMD training data not found at {PROCESSED_PATH}. Run `python scripts/download_imd.py` first."
        )

    frame = pd.read_csv(PROCESSED_PATH)
    normalized = normalize_observations(frame)
    return normalized, f"processed:{PROCESSED_PATH.relative_to(PROJECT_ROOT)}"


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

    # Ensure INSAT columns exist (NaN for regions without satellite coverage)
    for insat_col in ["insat_lst_mean_K", "insat_sst_mean_K"]:
        if insat_col not in normalized.columns:
            normalized[insat_col] = np.nan
        else:
            normalized[insat_col] = pd.to_numeric(normalized[insat_col], errors="coerce")

    normalized = normalized.dropna(subset=list(required_columns)).sort_values(["region", "date"]).reset_index(drop=True)
    return normalized
