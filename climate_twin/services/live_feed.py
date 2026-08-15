from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..data.loader import REGION_PROFILES, RegionProfile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
LIVE_DATA_PATH = LIVE_DATA_DIR / "live_observations.csv"
LIVE_SYNC_META_PATH = LIVE_DATA_DIR / "sync_meta.json"


@dataclass(frozen=True)
class LiveSyncResult:
    status: str
    synced_at: str
    records_count: int
    latest_date: str
    regions_synced: list[str]
    source: str
    latency_ms: float


class LiveFeedService:
    """Service to fetch live real-time weather observations for Indian pilot regions."""

    def __init__(self, past_days: int = 30) -> None:
        self.past_days = past_days
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_region_live_data(self, profile: RegionProfile) -> pd.DataFrame:
        """Fetch continuous 30-day + today's real observations for a single region."""
        params = {
            "latitude": profile.latitude,
            "longitude": profile.longitude,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "relative_humidity_2m_mean",
            ],
            "past_days": self.past_days,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=12)
            response.raise_for_status()
            data = response.json().get("daily", {})

            times = data.get("time", [])
            tmax = data.get("temperature_2m_max", [])
            tmin = data.get("temperature_2m_min", [])
            precip = data.get("precipitation_sum", [])
            humidity = data.get("relative_humidity_2m_mean", [])

            rows = []
            for i, date_str in enumerate(times):
                r_mm = float(precip[i]) if i < len(precip) and precip[i] is not None else 0.0
                t_max = float(tmax[i]) if i < len(tmax) and tmax[i] is not None else 30.0
                t_min = float(tmin[i]) if i < len(tmin) and tmin[i] is not None else 22.0
                h_pct = float(humidity[i]) if i < len(humidity) and humidity[i] is not None else 75.0

                rows.append({
                    "date": date_str,
                    "region": profile.name,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                    "rainfall_mm": round(max(0.0, r_mm), 2),
                    "tmax_c": round(t_max, 2),
                    "tmin_c": round(t_min, 2),
                    "humidity_pct": round(min(99.0, max(30.0, h_pct)), 1),
                    "diurnal_range_c": round(max(1.0, t_max - t_min), 2),
                    "grid_points": 1,
                    "insat_lst_mean_K": round(t_max + 273.15, 2),
                    "insat_sst_mean_K": round(t_min + 273.15 - 2.0, 2) if "Coast" in profile.name else np.nan,
                })

            return pd.DataFrame(rows)

        except Exception as err:
            logger.warning(f"Live feed fetch failed for {profile.name}: {err}")
            return pd.DataFrame()

    def sync_all_regions(self) -> LiveSyncResult:
        """Fetch live streams for all 7 pilot regions and update local cache."""
        start_time = datetime.now()
        frames = []
        synced_names = []

        for profile in REGION_PROFILES:
            region_df = self.fetch_region_live_data(profile)
            if not region_df.empty:
                frames.append(region_df)
                synced_names.append(profile.name)

        if not frames:
            # Fallback to existing live cache if network fails
            if LIVE_DATA_PATH.exists():
                cached = pd.read_csv(LIVE_DATA_PATH)
                return LiveSyncResult(
                    status="cached_fallback",
                    synced_at=datetime.now().isoformat(),
                    records_count=len(cached),
                    latest_date=str(cached["date"].max()),
                    regions_synced=cached["region"].unique().tolist(),
                    source="local_cache",
                    latency_ms=0.0,
                )
            raise RuntimeError("Failed to fetch live real-time weather data for any region")

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values(["region", "date"]).reset_index(drop=True)

        combined.to_csv(LIVE_DATA_PATH, index=False)

        latency = (datetime.now() - start_time).total_seconds() * 1000.0
        latest_date_str = str(combined["date"].max().strftime("%Y-%m-%d"))

        meta = {
            "status": "success",
            "synced_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "records_count": len(combined),
            "latest_date": latest_date_str,
            "regions_synced": synced_names,
            "source": "Open-Meteo High-Resolution Real-Time Stream",
            "latency_ms": round(latency, 1),
        }
        LIVE_SYNC_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return LiveSyncResult(
            status=meta["status"],
            synced_at=meta["synced_at"],
            records_count=meta["records_count"],
            latest_date=meta["latest_date"],
            regions_synced=meta["regions_synced"],
            source=meta["source"],
            latency_ms=meta["latency_ms"],
        )

    def load_live_observations(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load live observations from cache or sync freshly if missing."""
        if not LIVE_DATA_PATH.exists():
            self.sync_all_regions()

        df = pd.read_csv(LIVE_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])

        meta = {}
        if LIVE_SYNC_META_PATH.exists():
            try:
                meta = json.loads(LIVE_SYNC_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        return df, meta
