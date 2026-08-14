from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def extract_insat_to_csv(
    insat_dir: str | Path,
    output_path: str | Path,
    target_latitudes: np.ndarray | None = None,
    target_longitudes: np.ndarray | None = None,
    value_name_hint: str = "LST",
    years_back: int = 15,
) -> pd.DataFrame:
    """Extract INSAT L2B data into a daily CSV keyed by date, latitude, longitude."""

    insat_dir = Path(insat_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(insat_dir.rglob("*.h5")) + sorted(insat_dir.rglob("*.H5"))
    if not files:
        raise FileNotFoundError(f"No INSAT HDF5 files were found under {insat_dir}")

    years = _allowed_years_for_filter(years_back)
    filtered_files = [path for path in files if _file_year_in_scope(path, years)]
    if not filtered_files:
        raise ValueError(f"No INSAT HDF5 files found in the last {years_back} years under {insat_dir}")

    if target_latitudes is None or target_longitudes is None:
        sample_table = build_insat_daily_table(insat_dir, np.asarray([0.0]), np.asarray([0.0]), value_name_hint=value_name_hint)
        if sample_table.empty:
            raise ValueError("Could not infer grid coordinates from INSAT files.")
        target_latitudes = np.asarray(sorted(sample_table["latitude"].dropna().unique()))
        target_longitudes = np.asarray(sorted(sample_table["longitude"].dropna().unique()))

    extracted = build_insat_daily_table(
        insat_dir,
        np.asarray(target_latitudes),
        np.asarray(target_longitudes),
        value_name_hint=value_name_hint,
        files_to_use=filtered_files,
    )
    extracted.to_csv(output_file, index=False)
    LOGGER.info("Wrote extracted INSAT table to %s with %s rows", output_file, len(extracted))
    return extracted


def merge_insat_with_imd(
    imd_frame: pd.DataFrame,
    insat_dir: str | Path,
    output_path: str | Path | None = None,
    value_name_hint: str = "LST",
    years_back: int = 15,
) -> pd.DataFrame:
    """Merge INSAT L2B surface temperature onto an IMD fused daily dataframe.

    The function keeps the IMD frame as the base table and performs a left join on
    (date, latitude, longitude) so IMD rows are preserved when INSAT coverage is
    absent. This makes the supplement optional and keeps the main climate pipeline
    stable even when some INSAT files are missing.
    """

    frame = imd_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    if "latitude" not in frame.columns or "longitude" not in frame.columns:
        raise ValueError("IMD frame must contain latitude and longitude columns to merge INSAT data.")

    target_latitudes = np.asarray(sorted(frame["latitude"].dropna().unique()))
    target_longitudes = np.asarray(sorted(frame["longitude"].dropna().unique()))
    if target_latitudes.size == 0 or target_longitudes.size == 0:
        raise ValueError("IMD frame does not contain any valid latitude/longitude pairs for regridding.")

    insat_daily = build_insat_daily_table(Path(insat_dir), target_latitudes, target_longitudes, value_name_hint=value_name_hint)
    if insat_daily.empty:
        LOGGER.warning("No valid INSAT HDF5 files were found; keeping IMD data unchanged and adding NaN insat_lst_c.")
        frame["insat_lst_c"] = np.nan
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
        return frame

    merged = frame.merge(
        insat_daily[["date", "latitude", "longitude", "insat_lst_c"]],
        on=["date", "latitude", "longitude"],
        how="left",
    )

    valid_matches = int(merged["insat_lst_c"].notna().sum())
    missing_matches = int(merged["insat_lst_c"].isna().sum())
    LOGGER.info(
        "INSAT merge summary: %s rows matched valid data, %s rows are null (%0.2f%% coverage)",
        valid_matches,
        missing_matches,
        (valid_matches / len(merged) * 100.0) if len(merged) else 0.0,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged


def _allowed_years_for_filter(years_back: int) -> set[int]:
    current_year = pd.Timestamp.utcnow().year
    return set(range(current_year - years_back + 1, current_year + 1))


def _file_year_in_scope(file_path: Path, allowed_years: set[int]) -> bool:
    for year in sorted(allowed_years):
        if str(year) in file_path.name:
            return True
    return False


def build_insat_daily_table(
    insat_dir: Path,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
    value_name_hint: str = "LST",
    files_to_use: list[Path] | None = None,
) -> pd.DataFrame:
    """Read all INSAT HDF5 files in a folder and aggregate to a daily lat/lon table."""

    files = files_to_use if files_to_use is not None else sorted(insat_dir.rglob("*.h5")) + sorted(insat_dir.rglob("*.H5"))
    if not files:
        LOGGER.warning("No INSAT .h5 files were found under %s", insat_dir)
        return pd.DataFrame(columns=["date", "latitude", "longitude", "insat_lst_c"])

    daily_values: dict[pd.Timestamp, list[np.ndarray]] = {}
    daily_latitudes: dict[pd.Timestamp, np.ndarray] = {}
    daily_longitudes: dict[pd.Timestamp, np.ndarray] = {}

    for file_path in files:
        try:
            sample = load_insat_dataset(file_path, value_name_hint=value_name_hint)
        except Exception as exc:  # pragma: no cover - data-specific exceptions should be logged
            LOGGER.warning("Skipping %s: %s", file_path.name, exc)
            continue

        if sample is None:
            continue

        variable_name, data_array, lat_array, lon_array, timestamp = sample
        date_key = pd.Timestamp(timestamp).normalize()

        regridded = regrid_insat_to_target_grid(
            data_array=data_array,
            lat_array=lat_array,
            lon_array=lon_array,
            target_latitudes=target_latitudes,
            target_longitudes=target_longitudes,
        )

        daily_values.setdefault(date_key, []).append(regridded)
        daily_latitudes.setdefault(date_key, target_latitudes)
        daily_longitudes.setdefault(date_key, target_longitudes)

    rows: list[dict[str, Any]] = []
    for date_key, value_stack in daily_values.items():
        if not value_stack:
            continue

        aggregated = np.nanmean(np.stack(value_stack, axis=0), axis=0)
        lat_grid, lon_grid = np.meshgrid(target_latitudes, target_longitudes, indexing="ij")
        for idx_lat, latitude in enumerate(target_latitudes):
            for idx_lon, longitude in enumerate(target_longitudes):
                rows.append(
                    {
                        "date": date_key,
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                        "insat_lst_c": float(aggregated[idx_lat, idx_lon]),
                    }
                )

    table = pd.DataFrame(rows, columns=["date", "latitude", "longitude", "insat_lst_c"])
    if table.empty:
        LOGGER.warning("No valid INSAT values were assembled after regridding.")
        return table

    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    table["latitude"] = pd.to_numeric(table["latitude"], errors="coerce")
    table["longitude"] = pd.to_numeric(table["longitude"], errors="coerce")
    table = table.dropna(subset=["latitude", "longitude", "insat_lst_c"]).reset_index(drop=True)
    return table


def load_insat_dataset(file_path: Path, value_name_hint: str = "LST") -> tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp] | None:
    """Open one HDF5 file and pick a likely LST dataset plus its lat/lon arrays."""

    if not file_path.exists():
        raise FileNotFoundError(f"INSAT file does not exist: {file_path}")

    with h5py.File(file_path, "r") as handle:
        data_set_name, data_array = find_likely_value_dataset(handle, value_name_hint)
        lat_array = find_coordinate_array(handle, ["lat", "latitude", "y"])
        lon_array = find_coordinate_array(handle, ["lon", "longitude", "x"])

        if data_array is None:
            return None
        if lat_array is None or lon_array is None:
            raise ValueError(f"Could not locate lat/lon coordinates in {file_path.name}")

        timestamp = infer_timestamp_from_hdf(file_path, handle)
        return data_set_name, np.asarray(data_array, dtype=np.float32), np.asarray(lat_array), np.asarray(lon_array), timestamp


def find_likely_value_dataset(group: h5py.Group, value_name_hint: str) -> tuple[str, np.ndarray | None]:
    """Find a likely variable name for land-surface temperature among HDF5 groups."""

    candidates: list[tuple[str, Any]] = []
    for path, obj in walk_h5_groups(group):
        if not isinstance(obj, h5py.Dataset):
            continue
        name = str(obj.name).lower()
        if value_name_hint.lower() in name or "lst" in name or "land" in name and "surface" in name and "temperature" in name:
            candidates.append((path, np.asarray(obj)))

    if not candidates:
        return "", None

    # Prefer a 2D or 3D array that is not just a coordinate vector.
    for path, data in candidates:
        if data.ndim in {2, 3}:
            return path, data
    return candidates[0][0], candidates[0][1]


def find_coordinate_array(group: h5py.Group, preferred_names: list[str]) -> np.ndarray | None:
    """Locate a coordinate array in the HDF5 tree by likely names."""

    for path, obj in walk_h5_groups(group):
        if not isinstance(obj, h5py.Dataset):
            continue
        lower_name = str(obj.name).lower()
        if any(name in lower_name for name in preferred_names):
            return np.asarray(obj)
    return None


def walk_h5_groups(group: h5py.Group, prefix: str = "") -> list[tuple[str, Any]]:
    """Recursively iterate over all datasets in an HDF5 file."""

    results: list[tuple[str, Any]] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            results.append((path, value))
        elif isinstance(value, h5py.Group):
            results.extend(walk_h5_groups(value, path))
    return results


def infer_timestamp_from_hdf(file_path: Path, handle: h5py.File) -> pd.Timestamp:
    """Infer the observation timestamp from file name or HDF attributes."""

    filename = file_path.name
    patterns = [
        r"(\d{4})(\d{2})(\d{2})",
        r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})",
        r"(\d{8})T?(\d{6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                year, month, day = groups
                return pd.Timestamp(int(year), int(month), int(day))
            if len(groups) == 5:
                year, month, day, hour, minute = groups
                return pd.Timestamp(int(year), int(month), int(day), int(hour), int(minute))
            if len(groups) == 2:
                date_token, time_token = groups
                return pd.Timestamp(f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]} {time_token[:2]}:{time_token[2:4]}:{time_token[4:6]}")

    for key in ["start_time", "timestamp", "time", "date", "acquisition_time"]:
        if key in handle.attrs:
            value = handle.attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            try:
                return pd.to_datetime(value)
            except Exception:
                continue

    raise ValueError(f"Could not infer timestamp from INSAT file name or HDF attributes: {filename}")


def regrid_insat_to_target_grid(
    data_array: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
) -> np.ndarray:
    """Regrid a single INSAT scan onto the IMD target grid using nearest-neighbor matching."""

    values = np.asarray(data_array, dtype=np.float32)
    lat_values = np.asarray(lat_array)
    lon_values = np.asarray(lon_array)

    if lat_values.ndim == 1 and lon_values.ndim == 1 and values.shape == (lat_values.size, lon_values.size):
        source_lat, source_lon = np.meshgrid(lat_values, lon_values, indexing="ij")
        source_points = np.column_stack((source_lat.ravel(), source_lon.ravel()))
        source_values = values.ravel()
    elif lat_values.shape == values.shape and lon_values.shape == values.shape:
        source_points = np.column_stack((lat_values.ravel(), lon_values.ravel()))
        source_values = values.ravel()
    elif values.size == lat_values.size == lon_values.size:
        source_points = np.column_stack((lat_values.ravel(), lon_values.ravel()))
        source_values = values.ravel()
    else:
        raise ValueError(
            f"INSAT array shape {values.shape} is not compatible with coordinate lengths "
            f"{lat_values.size} and {lon_values.size}."
        )

    valid_mask = np.isfinite(source_values)
    if not valid_mask.any():
        return np.full((target_latitudes.size, target_longitudes.size), np.nan, dtype=np.float32)

    lat_grid, lon_grid = np.meshgrid(target_latitudes, target_longitudes, indexing="ij")
    regridded = np.full(lat_grid.shape, np.nan, dtype=np.float32)

    for idx_lat in range(lat_grid.shape[0]):
        for idx_lon in range(lat_grid.shape[1]):
            target_lat = float(lat_grid[idx_lat, idx_lon])
            target_lon = float(lon_grid[idx_lat, idx_lon])
            target_point = np.array([target_lat, target_lon], dtype=np.float64)
            valid_points = source_points[valid_mask]
            valid_values = source_values[valid_mask]
            distances = np.sum((valid_points - target_point) ** 2, axis=1)
            nearest_idx = int(np.argmin(distances))
            regridded[idx_lat, idx_lon] = float(valid_values[nearest_idx])

    return regridded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("INSAT merge helper loaded. Import this module instead of running it directly.")
