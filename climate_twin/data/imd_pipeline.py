from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = Path.home() / "Downloads"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "imd_15y"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "climate_training_data.csv"

DEFAULT_START_YEAR = 2011
DEFAULT_END_YEAR = 2025


@dataclass(frozen=True)
class RegionProfile:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class GrdSource:
    name: str
    page_url: str
    endpoint: str
    form_field: str
    value_column: str
    file_prefix: str
    rows: int
    cols: int
    lat_start: float
    lon_start: float
    lat_step: float
    lon_step: float
    local_patterns: tuple[str, ...]


REGION_PROFILES = [
    RegionProfile("Kerala Coast", 9.5, 76.5),
    RegionProfile("Indo-Gangetic Plain", 26.5, 81.0),
    RegionProfile("Northeast", 26.0, 91.7),
    RegionProfile("Central India", 22.0, 79.0),
    RegionProfile("Deccan Plateau", 17.8, 78.5),
    RegionProfile("Rajasthan Desert", 27.0, 73.0),
    RegionProfile("Coastal Odisha", 20.5, 85.5),
]

SOURCES = {
    "rainfall": GrdSource(
        name="rainfall",
        page_url="https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
        endpoint="rainfall.php",
        form_field="rain",
        value_column="rainfall_mm",
        file_prefix="rainfall",
        rows=129,
        cols=135,
        lat_start=6.5,
        lon_start=66.5,
        lat_step=0.25,
        lon_step=0.25,
        local_patterns=(
            "Rainfall_ind{year}_rfp25.grd",
            "Rainfall_ind{year}_rfp25 (1).grd",
            "rainfall_ind{year}_rfp25.grd",
            "ind{year}_rfp25.grd",
            "ind{year}_rfp25 (1).grd",
        ),
    ),
    "tmax": GrdSource(
        name="tmax",
        page_url="https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
        endpoint="maxtemp.php",
        form_field="maxtemp",
        value_column="tmax_c",
        file_prefix="tmax",
        rows=31,
        cols=31,
        lat_start=7.5,
        lon_start=67.5,
        lat_step=1.0,
        lon_step=1.0,
        local_patterns=(
            "Maxtemp_MaxT_{year}.GRD",
            "Maxtemp_MaxT_{year} (1).GRD",
            "Maxtemp_MaxT_{year}.grd",
            "max_temp_{year}.grd",
        ),
    ),
    "tmin": GrdSource(
        name="tmin",
        page_url="https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
        endpoint="mintemp.php",
        form_field="mintemp",
        value_column="tmin_c",
        file_prefix="tmin",
        rows=31,
        cols=31,
        lat_start=7.5,
        lon_start=67.5,
        lat_step=1.0,
        lon_step=1.0,
        local_patterns=(
            "Mintemp_MinT_{year}.GRD",
            "Mintemp_MinT_{year} (1).GRD",
            "Mintemp_MinT_{year}.grd",
            "min_temp_{year}.grd",
        ),
    ),
}


def nearest_region(latitude: float, longitude: float) -> RegionProfile:
    return min(
        REGION_PROFILES,
        key=lambda region: (region.latitude - latitude) ** 2 + (region.longitude - longitude) ** 2,
    )


def nearest_grid_index(source: GrdSource, latitude: float, longitude: float) -> tuple[int, int]:
    row = int(round((latitude - source.lat_start) / source.lat_step))
    col = int(round((longitude - source.lon_start) / source.lon_step))
    row = max(0, min(source.rows - 1, row))
    col = max(0, min(source.cols - 1, col))
    return row, col


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def expected_record_count(source: GrdSource, year: int) -> int:
    days = 366 if is_leap_year(year) else 365
    return days * source.rows * source.cols


def _download_bytes(source: GrdSource, year: int) -> bytes:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": source.page_url,
    }
    payload = {source.form_field: str(year)}
    download_url = _join_endpoint(source.page_url, source.endpoint)

    try:
        response = session.post(download_url, data=payload, headers=headers, timeout=120)
    except requests.exceptions.SSLError:
        response = session.post(download_url, data=payload, headers=headers, timeout=120, verify=False)

    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError(f"IMD response for {source.name} {year} was HTML, not binary data")
    return response.content


def _join_endpoint(page_url: str, endpoint: str) -> str:
    return page_url.rsplit("/", 1)[0] + "/" + endpoint


def find_local_source_file(source: GrdSource, year: int) -> Path | None:
    candidates: list[Path] = []
    for pattern in source.local_patterns:
        candidates.extend(DOWNLOAD_DIR.glob(pattern.format(year=year)))
        candidates.extend(RAW_CACHE_DIR.glob(f"**/{pattern.format(year=year)}"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def ensure_grd_file(source: GrdSource, year: int, force_download: bool = False) -> Path:
    cache_dir = RAW_CACHE_DIR / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source.file_prefix}_{year}.grd"

    if cache_path.exists() and not force_download:
        return cache_path

    local_source = find_local_source_file(source, year)
    if local_source is not None and not force_download:
        shutil.copy2(local_source, cache_path)
        return cache_path

    content = _download_bytes(source, year)
    cache_path.write_bytes(content)
    return cache_path


def read_grd_cube(source: GrdSource, grd_path: Path, year: int) -> np.ndarray:
    days = 366 if is_leap_year(year) else 365
    expected_values = expected_record_count(source, year)
    cube = np.fromfile(grd_path, dtype="<f4", count=expected_values)
    if cube.size != expected_values:
        raise ValueError(f"{grd_path.name} has {cube.size} values, expected {expected_values}")
    return cube.reshape((days, source.rows, source.cols))


def _day_range(year: int) -> Iterable[date]:
    days = 366 if is_leap_year(year) else 365
    for offset in range(days):
        yield date(year, 1, 1) + timedelta(days=offset)


def convert_grd_to_csv(
    source: GrdSource,
    year: int,
    grd_path: Path,
    csv_path: Path,
    mode: str = "regional",
) -> pd.DataFrame:
    cube = read_grd_cube(source, grd_path, year)

    if mode not in {"regional", "full-grid"}:
        raise ValueError("mode must be 'regional' or 'full-grid'")

    if mode == "full-grid":
        lat_values = source.lat_start + np.arange(source.rows) * source.lat_step
        lon_values = source.lon_start + np.arange(source.cols) * source.lon_step
        lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing="ij")
        records: list[dict[str, object]] = []
        for day_index, current_date in enumerate(_day_range(year)):
            day_values = cube[day_index].astype(float).reshape(-1)
            frame = pd.DataFrame(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "latitude": lat_grid.reshape(-1),
                    "longitude": lon_grid.reshape(-1),
                    source.value_column: day_values,
                }
            )
            records.append(frame)

        merged = pd.concat(records, ignore_index=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(csv_path, index=False)
        return merged

    records: list[dict[str, object]] = []
    for region in REGION_PROFILES:
        row_index, col_index = nearest_grid_index(source, region.latitude, region.longitude)
        series = cube[:, row_index, col_index]
        for current_date, value in zip(_day_range(year), series, strict=True):
            records.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "region": region.name,
                    "latitude": region.latitude,
                    "longitude": region.longitude,
                    source.value_column: float(value),
                    "grid_row": row_index,
                    "grid_col": col_index,
                }
            )

    frame = pd.DataFrame(records).sort_values(["region", "date"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return frame


def build_yearly_source_csvs(start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR, mode: str = "regional", force_download: bool = False) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {name: [] for name in SOURCES}
    for year in range(start_year, end_year + 1):
        for source_name, source in SOURCES.items():
            grd_path = ensure_grd_file(source, year, force_download=force_download)
            csv_dir = RAW_CACHE_DIR / source.name / "csv"
            csv_path = csv_dir / f"{source.file_prefix}_{year}.csv"
            convert_grd_to_csv(source, year, grd_path, csv_path, mode=mode)
            outputs[source_name].append(csv_path)
    return outputs


def build_processed_training_table(start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR, mode: str = "regional", force_download: bool = False) -> Path:
    build_yearly_source_csvs(start_year=start_year, end_year=end_year, mode=mode, force_download=force_download)

    rainfall_frames: list[pd.DataFrame] = []
    tmax_frames: list[pd.DataFrame] = []
    tmin_frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        for source_name, frames in (("rainfall", rainfall_frames), ("tmax", tmax_frames), ("tmin", tmin_frames)):
            csv_path = RAW_CACHE_DIR / source_name / "csv" / f"{source_name}_{year}.csv"
            if not csv_path.exists():
                continue
            frames.append(pd.read_csv(csv_path))

    if not rainfall_frames or not tmax_frames or not tmin_frames:
        raise FileNotFoundError("One or more IMD source groups could not be built; check downloads and file paths.")

    rainfall = pd.concat(rainfall_frames, ignore_index=True)
    tmax = pd.concat(tmax_frames, ignore_index=True)
    tmin = pd.concat(tmin_frames, ignore_index=True)

    merged = rainfall.merge(tmax, on=["date", "region", "latitude", "longitude"], how="inner").merge(
        tmin, on=["date", "region", "latitude", "longitude"], how="inner"
    )

    merged = merged.dropna().sort_values(["region", "date"]).reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged["diurnal_range_c"] = merged["tmax_c"] - merged["tmin_c"]
    merged["humidity_pct"] = np.clip(92.0 - merged["diurnal_range_c"] * 2.0 + np.log1p(merged["rainfall_mm"].clip(lower=0)) * 3.0, 35.0, 98.0)
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["month"] = merged["date"].dt.month
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12.0)
    merged["rain_intensity"] = np.log1p(merged["rainfall_mm"].clip(lower=0))
    merged["temperature_gap"] = merged["tmax_c"] - merged["tmin_c"]
    merged["grid_points"] = 1
    merged["region_code"] = merged["region"].astype("category").cat.codes
    merged = merged.drop(columns=["month"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download IMD GRD files for the previous 15 years and build the processed climate table.")
    parser.add_argument("--start", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--mode", choices=["regional", "full-grid"], default="regional")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    output = build_processed_training_table(
        start_year=args.start,
        end_year=args.end,
        mode=args.mode,
        force_download=args.force_download,
    )
    print(f"Saved processed training table to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())