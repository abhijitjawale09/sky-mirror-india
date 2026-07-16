from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegionProfile:
    name: str
    latitude: float
    longitude: float
    rainfall_bias: float
    temperature_bias: float


REGION_PROFILES = [
    RegionProfile("Kerala Coast", 9.5, 76.5, 1.45, -0.5),
    RegionProfile("Indo-Gangetic Plain", 26.5, 81.0, 0.9, 1.4),
    RegionProfile("Northeast", 26.0, 91.7, 1.6, -0.2),
    RegionProfile("Central India", 22.0, 79.0, 1.0, 0.8),
    RegionProfile("Deccan Plateau", 17.8, 78.5, 0.78, 1.0),
    RegionProfile("Rajasthan Desert", 27.0, 73.0, 0.28, 2.0),
    RegionProfile("Coastal Odisha", 20.5, 85.5, 1.18, 0.6),
]


def get_region_profile(region_name: str) -> RegionProfile:
    for profile in REGION_PROFILES:
        if profile.name == region_name:
            return profile
    return REGION_PROFILES[0]


def build_demo_observations(days: int = 540) -> pd.DataFrame:
    end_date = datetime.now(UTC).date()
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    rows: list[dict[str, float | str | datetime]] = []

    for region_index, profile in enumerate(REGION_PROFILES):
        seasonal_phase = region_index * 0.37
        for day_index, current_date in enumerate(dates):
            annual_wave = np.sin(2 * np.pi * day_index / 365.25 + seasonal_phase)
            monsoon_wave = np.sin(2 * np.pi * day_index / 120.0 + seasonal_phase * 0.5)
            heat_wave = np.cos(2 * np.pi * day_index / 180.0 + seasonal_phase)
            noise = np.random.default_rng(region_index * 10_000 + day_index).normal(0, 0.35)

            rainfall = max(0.1, 3.0 * profile.rainfall_bias * (1.0 + 0.8 * monsoon_wave) + 2.0 * annual_wave + noise)
            tmax = 29.5 + profile.temperature_bias + 6.5 * heat_wave + 2.5 * annual_wave + noise * 0.5
            tmin = tmax - (6.0 + 0.8 * np.cos(2 * np.pi * day_index / 30.0 + seasonal_phase))

            rows.append(
                {
                    "date": current_date,
                    "region": profile.name,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                    "rainfall_mm": round(float(rainfall), 2),
                    "tmax_c": round(float(tmax), 2),
                    "tmin_c": round(float(tmin), 2),
                    "humidity_pct": round(float(np.clip(70 + 12 * monsoon_wave - 3 * heat_wave + noise * 2, 35, 98)), 2),
                }
            )

    return pd.DataFrame(rows)


def load_training_observations() -> tuple[pd.DataFrame, str]:
    project_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        project_root / "data" / "raw" / "imd" / "processed" / "climate_observations.csv",
        project_root / "data" / "raw" / "imd" / "processed" / "climate_observations.parquet",
        project_root / "data" / "raw" / "imd" / "climate_observations.csv",
        project_root / "data" / "raw" / "imd" / "climate_observations.parquet",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)

        normalized = normalize_observations(frame)
        return normalized, f"local:{path.relative_to(project_root)}"

    return build_demo_observations(), "synthetic_demo"


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

    normalized = normalized.dropna(subset=list(required_columns)).sort_values(["region", "date"]).reset_index(drop=True)
    return normalized
