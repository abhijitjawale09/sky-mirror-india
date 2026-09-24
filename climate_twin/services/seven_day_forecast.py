"""Seven-day climate forecast service using the Open-Meteo Forecast API.

This module provides structured 7-day daily forecast data including
temperature extremes, precipitation, humidity, wind, UV, and derived
weather conditions with confidence indicators.

The service is designed so the underlying API can be replaced without
changing the return contract (list of ``DayForecast`` dicts).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

from ..data.loader import REGION_PROFILES, RegionProfile

logger = logging.getLogger(__name__)

# Open-Meteo API (same base as existing LiveFeedService)
_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables requested from the API
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "uv_index_max",
    "weather_code",
]

# WMO Weather Interpretation Codes → human label + emoji icon
_WMO_CONDITIONS: dict[int, tuple[str, str]] = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "🌥️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Dense Drizzle", "🌧️"),
    56: ("Freezing Drizzle", "🌧️"),
    57: ("Dense Freezing Drizzle", "🌧️"),
    61: ("Slight Rain", "🌦️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    66: ("Freezing Rain", "🌧️"),
    67: ("Heavy Freezing Rain", "🌧️"),
    71: ("Slight Snow", "🌨️"),
    73: ("Moderate Snow", "🌨️"),
    75: ("Heavy Snow", "❄️"),
    77: ("Snow Grains", "❄️"),
    80: ("Slight Rain Showers", "🌦️"),
    81: ("Moderate Rain Showers", "🌧️"),
    82: ("Violent Rain Showers", "⛈️"),
    85: ("Slight Snow Showers", "🌨️"),
    86: ("Heavy Snow Showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm + Hail", "🌩️"),
    99: ("Thunderstorm + Heavy Hail", "🌩️"),
}

# Wind direction cardinal labels
_WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _degrees_to_cardinal(degrees: float | None) -> str:
    """Convert wind direction in degrees to a 16-point compass label."""
    if degrees is None:
        return "—"
    idx = round(degrees / 22.5) % 16
    return _WIND_DIRS[idx]


def _confidence_for_day(day_index: int) -> dict[str, str]:
    """Return confidence level and CSS class based on forecast horizon distance.

    Days 1-2 are High, 3-5 are Medium, 6-7 are Low — mirroring the
    characteristic skill decay of NWP and statistical models.
    """
    if day_index < 2:
        return {"level": "High", "class": "confidence-high"}
    if day_index < 5:
        return {"level": "Medium", "class": "confidence-medium"}
    return {"level": "Low", "class": "confidence-low"}


def _condition_from_code(code: int | None) -> tuple[str, str]:
    """Derive a human-readable condition label + icon from WMO weather code."""
    if code is None:
        return ("Unknown", "❓")
    return _WMO_CONDITIONS.get(code, ("Unknown", "❓"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return default


class SevenDayForecastService:
    """Fetch and structure 7-day daily forecasts from Open-Meteo."""

    def __init__(self) -> None:
        self.base_url = _BASE_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        region_name: str | None = None,
    ) -> dict[str, Any]:
        """Return a 7-day forecast payload for the given coordinates.

        Parameters
        ----------
        latitude, longitude:
            Decimal degree coordinates of the location.
        region_name:
            Optional display name; auto-resolved from pilot regions
            when not provided.

        Returns
        -------
        dict with keys:
            location      – resolved name or "Custom Location"
            latitude      – echo-back
            longitude     – echo-back
            last_updated  – ISO timestamp string
            days          – list of 7 DayForecast dicts
            error         – None on success, error message on failure
        """
        location_label = region_name or self._resolve_region_name(latitude, longitude)

        try:
            raw = self._fetch_api(latitude, longitude)
        except Exception as exc:
            logger.warning("7-day forecast fetch failed: %s", exc)
            return {
                "location": location_label,
                "latitude": latitude,
                "longitude": longitude,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
                "days": [],
                "error": str(exc),
            }

        days = self._parse_days(raw)

        return {
            "location": location_label,
            "latitude": latitude,
            "longitude": longitude,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            "days": days,
            "error": None,
        }

    def get_forecast_for_region(self, region_name: str) -> dict[str, Any]:
        """Convenience: look up pilot region coordinates and fetch forecast."""
        profile = self._find_profile(region_name)
        if profile is None:
            return {
                "location": region_name,
                "latitude": None,
                "longitude": None,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
                "days": [],
                "error": f"Unknown region: {region_name}",
            }
        return self.get_forecast(profile.latitude, profile.longitude, profile.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_api(self, lat: float, lon: float) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": _DAILY_VARS,
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(self.base_url, params=params, timeout=12)
        resp.raise_for_status()
        return resp.json().get("daily", {})

    def _parse_days(self, raw: dict) -> list[dict[str, Any]]:
        times = raw.get("time", [])
        results: list[dict[str, Any]] = []

        for i, date_str in enumerate(times[:7]):
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            code = raw.get("weather_code", [None] * 7)
            wmo = code[i] if i < len(code) else None
            condition_label, condition_icon = _condition_from_code(wmo)
            confidence = _confidence_for_day(i)

            tmax = _safe_float(self._idx(raw, "temperature_2m_max", i), 30.0)
            tmin = _safe_float(self._idx(raw, "temperature_2m_min", i), 22.0)
            precip = _safe_float(self._idx(raw, "precipitation_sum", i), 0.0)
            precip_prob = _safe_float(self._idx(raw, "precipitation_probability_max", i), 0.0)
            humidity = _safe_float(self._idx(raw, "relative_humidity_2m_mean", i), 60.0)
            wind_speed = _safe_float(self._idx(raw, "wind_speed_10m_max", i), 0.0)
            wind_deg = self._idx(raw, "wind_direction_10m_dominant", i)
            uv = _safe_float(self._idx(raw, "uv_index_max", i), 0.0)

            results.append({
                "date": date_str,
                "day_name": dt.strftime("%A"),
                "day_short": dt.strftime("%a"),
                "date_display": dt.strftime("%d %b"),
                "condition": condition_label,
                "condition_icon": condition_icon,
                "tmax_c": tmax,
                "tmin_c": tmin,
                "rainfall_mm": precip,
                "rainfall_probability_pct": precip_prob,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind_speed,
                "wind_direction": _degrees_to_cardinal(wind_deg),
                "uv_index": uv,
                "confidence": confidence,
                "basis": "model_forecast",
            })

        return results

    @staticmethod
    def _idx(mapping: dict, key: str, index: int) -> Any:
        arr = mapping.get(key, [])
        return arr[index] if index < len(arr) else None

    @staticmethod
    def _find_profile(name: str) -> RegionProfile | None:
        for p in REGION_PROFILES:
            if p.name.lower() == name.lower():
                return p
        return None

    @staticmethod
    def _resolve_region_name(lat: float, lon: float) -> str:
        """Attempt to match coordinates to a known pilot region."""
        for p in REGION_PROFILES:
            if abs(p.latitude - lat) < 0.6 and abs(p.longitude - lon) < 0.6:
                return p.name
        return "Custom Location"
