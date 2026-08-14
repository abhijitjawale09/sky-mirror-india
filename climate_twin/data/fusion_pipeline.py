from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_IMD_DIR = PROJECT_ROOT / "data" / "raw" / "imd_15y"
RAW_INSAT_DIR = PROJECT_ROOT / "data" / "raw" / "insat"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FUSED_DIR = PROJECT_ROOT / "data" / "fused"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

COMMON_GRID_STEP = 0.25


def list_imd_files() -> list[Path]:
    """Return all IMD source files found in the project."""
    files = []
    for root in [PROJECT_ROOT / "data" / "raw" / "imd", RAW_IMD_DIR]:
        if root.exists():
            files.extend(sorted(root.rglob("*.csv")))
    return sorted(set(files))


def list_insat_files() -> list[Path]:
    """Return all INSAT source files found in the project."""
    insat_dir = RAW_INSAT_DIR
    if not insat_dir.exists():
        return []
    return sorted(insat_dir.glob("*.csv"))


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect a CSV dataset and return schema, time, and spatial metadata."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)

    def first_present(columns: list[str]) -> str | None:
        for column in columns:
            if column in frame.columns:
                return column
        return None

    candidate_date = first_present(["date", "observation_date", "timestamp", "observation_time"])
    candidate_lat = first_present(["latitude", "latitude_deg", "lat", "y"])
    candidate_lon = first_present(["longitude", "longitude_deg", "lon", "x"])
    candidate_rain = first_present(["rainfall_mm", "rainfall_accumulation_mm", "precipitation_mm"])
    candidate_tmax = first_present(["tmax_c", "temperature_max_c", "max_temperature_c", "land_surface_temperature_K"])
    candidate_tmin = first_present(["tmin_c", "temperature_min_c", "min_temperature_c"])
    candidate_lsts = first_present(["land_surface_temperature_K", "lst_k", "land_surface_temperature_c"])
    candidate_sst = first_present(["sea_surface_temperature_K", "sst_k", "sea_surface_temp_k"])

    date_values = pd.to_datetime(frame[candidate_date], errors="coerce") if candidate_date else pd.Series(pd.NaT, index=frame.index)
    lat_values = pd.to_numeric(frame[candidate_lat], errors="coerce") if candidate_lat else pd.Series(pd.NaN, index=frame.index)
    lon_values = pd.to_numeric(frame[candidate_lon], errors="coerce") if candidate_lon else pd.Series(pd.NaN, index=frame.index)

    summary = {
        "path": str(csv_path),
        "file_name": csv_path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "date_column": candidate_date,
        "latitude_column": candidate_lat,
        "longitude_column": candidate_lon,
        "rainfall_column": candidate_rain,
        "max_temperature_column": candidate_tmax,
        "min_temperature_column": candidate_tmin,
        "lst_column": candidate_lsts,
        "sst_column": candidate_sst,
        "date_range": [
            (date_values.min().strftime("%Y-%m-%d") if date_values.notna().any() else None),
            (date_values.max().strftime("%Y-%m-%d") if date_values.notna().any() else None),
        ],
        "latitude_range": [
            float(lat_values.min()) if lat_values.notna().any() else None,
            float(lat_values.max()) if lat_values.notna().any() else None,
        ],
        "longitude_range": [
            float(lon_values.min()) if lon_values.notna().any() else None,
            float(lon_values.max()) if lon_values.notna().any() else None,
        ],
    }

    if lat_values.notna().any() and lon_values.notna().any():
        unique_lat = sorted(np.unique(np.round(lat_values.dropna().to_numpy() / COMMON_GRID_STEP) * COMMON_GRID_STEP))
        unique_lon = sorted(np.unique(np.round(lon_values.dropna().to_numpy() / COMMON_GRID_STEP) * COMMON_GRID_STEP))
        summary["common_grid_latitudes"] = unique_lat[:10], "..."
        summary["common_grid_longitudes"] = unique_lon[:10], "..."
        summary["grid_resolution_deg"] = COMMON_GRID_STEP
    else:
        summary["grid_resolution_deg"] = None

    return summary


def load_imd_data() -> pd.DataFrame:
    """Load the raw IMD daily gridded CSVs and standardize them.

    This project contains IMD rainfall, maximum temperature, and minimum temperature
    products as CSV exports with daily latitude/longitude cells. The pipeline keeps
    the raw files untouched and works from them directly.
    """
    frames: list[pd.DataFrame] = []
    for path in list_imd_files():
        if path.name.lower().startswith("climate_observations"):
            continue
        df = pd.read_csv(path)
        df = df.copy()
        df["source_file"] = path.name

        if {"date", "latitude", "longitude"}.issubset(df.columns):
            df = df.rename(columns={"date": "observation_date", "latitude": "latitude_deg", "longitude": "longitude_deg"})
            if "rainfall_mm" in df.columns:
                df["imd_rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
            elif "rainfall_accumulation_mm" in df.columns:
                df["imd_rainfall_mm"] = pd.to_numeric(df["rainfall_accumulation_mm"], errors="coerce")
            if "tmax_c" in df.columns:
                df["imd_max_temperature_C"] = pd.to_numeric(df["tmax_c"], errors="coerce")
            if "tmin_c" in df.columns:
                df["imd_min_temperature_C"] = pd.to_numeric(df["tmin_c"], errors="coerce")

        elif {"date", "latitude", "longitude", "rainfall_mm"}.issubset(df.columns):
            df = df.rename(columns={"date": "observation_date", "latitude": "latitude_deg", "longitude": "longitude_deg"})
            df["imd_rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

        elif {"date", "latitude", "longitude", "tmax_c"}.issubset(df.columns):
            df = df.rename(columns={"date": "observation_date", "latitude": "latitude_deg", "longitude": "longitude_deg"})
            df["imd_max_temperature_C"] = pd.to_numeric(df["tmax_c"], errors="coerce")

        elif {"date", "latitude", "longitude", "tmin_c"}.issubset(df.columns):
            df = df.rename(columns={"date": "observation_date", "latitude": "latitude_deg", "longitude": "longitude_deg"})
            df["imd_min_temperature_C"] = pd.to_numeric(df["tmin_c"], errors="coerce")

        else:
            continue

        df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
        numeric_cols = ["latitude_deg", "longitude_deg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No IMD CSV datasets found under {RAW_IMD_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["observation_date", "latitude_deg", "longitude_deg"], keep="last")
    combined["latitude_deg"] = np.round(combined["latitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    combined["longitude_deg"] = np.round(combined["longitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    return combined.reset_index(drop=True)


def load_insat_data() -> pd.DataFrame:
    """Load and standardize all INSAT CSV files found in the project."""
    frames: list[pd.DataFrame] = []
    for path in list_insat_files():
        df = pd.read_csv(path)
        df = df.copy()
        rename_map = {
            "observation_date": "observation_date",
            "latitude_deg": "latitude_deg",
            "longitude_deg": "longitude_deg",
            "land_surface_temperature_K": "insat_lst_K",
            "rainfall_accumulation_mm": "insat_rainfall_mm",
            "sea_surface_temperature_K": "insat_sst_K",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "insat_lst_K" in df.columns:
            df["insat_lst_K"] = pd.to_numeric(df["insat_lst_K"], errors="coerce")
        if "insat_rainfall_mm" in df.columns:
            df["insat_rainfall_mm"] = pd.to_numeric(df["insat_rainfall_mm"], errors="coerce")
        if "insat_sst_K" in df.columns:
            df["insat_sst_K"] = pd.to_numeric(df["insat_sst_K"], errors="coerce")
        df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
        if "latitude_deg" in df.columns:
            df["latitude_deg"] = pd.to_numeric(df["latitude_deg"], errors="coerce")
        if "longitude_deg" in df.columns:
            df["longitude_deg"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
        df = df.dropna(subset=["observation_date", "latitude_deg", "longitude_deg"])
        df["latitude_deg"] = np.round(df["latitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
        df["longitude_deg"] = np.round(df["longitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No INSAT CSV datasets found under {RAW_INSAT_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["observation_date", "latitude_deg", "longitude_deg"], keep="last")
    return combined.reset_index(drop=True)


def clean_imd_data(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Clean IMD grid cells, keep only actual values, and align to a 0.25° common grid."""
    frame = raw_frame.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    for column in ["latitude_deg", "longitude_deg"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["latitude_deg"] = np.round(frame["latitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    frame["longitude_deg"] = np.round(frame["longitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP

    if "imd_rainfall_mm" in frame.columns:
        frame["imd_rainfall_mm"] = pd.to_numeric(frame["imd_rainfall_mm"], errors="coerce")
    if "imd_max_temperature_C" in frame.columns:
        frame["imd_max_temperature_C"] = pd.to_numeric(frame["imd_max_temperature_C"], errors="coerce")
    if "imd_min_temperature_C" in frame.columns:
        frame["imd_min_temperature_C"] = pd.to_numeric(frame["imd_min_temperature_C"], errors="coerce")

    frame = frame.dropna(subset=["observation_date", "latitude_deg", "longitude_deg"]).reset_index(drop=True)
    return frame


def clean_insat_data(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Clean INSAT observations, preserving actual measured values and keeping missing values explicit."""
    frame = raw_frame.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["latitude_deg"] = pd.to_numeric(frame.get("latitude_deg", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["longitude_deg"] = pd.to_numeric(frame.get("longitude_deg", pd.Series(np.nan, index=frame.index)), errors="coerce")

    for column in ["insat_lst_K", "insat_rainfall_mm", "insat_sst_K"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["latitude_deg"] = np.round(frame["latitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    frame["longitude_deg"] = np.round(frame["longitude_deg"] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    frame = frame.dropna(subset=["observation_date", "latitude_deg", "longitude_deg"]).reset_index(drop=True)
    return frame


def harmonize_time(frame: pd.DataFrame, date_column: str = "observation_date") -> pd.DataFrame:
    """Normalize all timestamps to a daily date column without inventing new dates."""
    harmonized = frame.copy()
    harmonized[date_column] = pd.to_datetime(harmonized[date_column], errors="coerce")
    harmonized[date_column] = harmonized[date_column].dt.normalize()
    return harmonized


def harmonize_spatial_grid(frame: pd.DataFrame, lat_col: str = "latitude_deg", lon_col: str = "longitude_deg") -> pd.DataFrame:
    """Align both IMD and INSAT datasets to a common 0.25° analysis grid.

    The source datasets do not share an identical grid, so each coordinate is mapped to the
    nearest 0.25° cell rather than being silently dropped or arbitrarily imputed.
    """
    harmonized = frame.copy()
    harmonized[lat_col] = np.round(harmonized[lat_col] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    harmonized[lon_col] = np.round(harmonized[lon_col] / COMMON_GRID_STEP) * COMMON_GRID_STEP
    return harmonized


def aggregate_insat_daily(insat_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate INSAT observations from the raw daily points into a daily grid table."""
    frame = harmonize_time(insat_frame, "observation_date").copy()
    frame = harmonize_spatial_grid(frame, "latitude_deg", "longitude_deg")

    grouped = (
        frame.groupby(["observation_date", "latitude_deg", "longitude_deg"], as_index=False)
        .agg(
            insat_lst_mean_K=("insat_lst_K", "mean"),
            insat_lst_min_K=("insat_lst_K", "min"),
            insat_lst_max_K=("insat_lst_K", "max"),
            insat_rainfall_daily_mm=("insat_rainfall_mm", "sum"),
            insat_sst_mean_K=("insat_sst_K", "mean"),
        )
    )

    grouped["insat_lst_mean_K"] = grouped["insat_lst_mean_K"].apply(lambda value: np.nan if pd.isna(value) else float(value))
    grouped["insat_lst_min_K"] = grouped["insat_lst_min_K"].apply(lambda value: np.nan if pd.isna(value) else float(value))
    grouped["insat_lst_max_K"] = grouped["insat_lst_max_K"].apply(lambda value: np.nan if pd.isna(value) else float(value))
    grouped["insat_rainfall_daily_mm"] = grouped["insat_rainfall_daily_mm"].apply(lambda value: np.nan if pd.isna(value) else float(value))
    grouped["insat_sst_mean_K"] = grouped["insat_sst_mean_K"].apply(lambda value: np.nan if pd.isna(value) else float(value))
    return grouped.reset_index(drop=True)


def merge_imd_insat(imd_frame: pd.DataFrame, insat_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge IMD daily grid cells with daily INSAT values using date + lat/lon.

    Data are matched on the harmonized daily key, not on raw row number or a blind exact timestamp join.
    """
    imd = harmonize_time(imd_frame, "observation_date").copy()
    imd = harmonize_spatial_grid(imd, "latitude_deg", "longitude_deg")
    insat = aggregate_insat_daily(insat_frame)

    imd_key_cols = ["observation_date", "latitude_deg", "longitude_deg"]
    imd = imd.groupby(imd_key_cols, as_index=False).agg({
        "imd_rainfall_mm": "mean",
        "imd_max_temperature_C": "mean",
        "imd_min_temperature_C": "mean",
    })

    fused = imd.merge(
        insat,
        on=["observation_date", "latitude_deg", "longitude_deg"],
        how="left",
    )

    fused = fused.rename(columns={
        "observation_date": "observation_date",
        "latitude_deg": "latitude_deg",
        "longitude_deg": "longitude_deg",
    })

    if "imd_max_temperature_C" in fused.columns:
        fused["imd_max_temperature_C"] = pd.to_numeric(fused["imd_max_temperature_C"], errors="coerce")
    if "imd_min_temperature_C" in fused.columns:
        fused["imd_min_temperature_C"] = pd.to_numeric(fused["imd_min_temperature_C"], errors="coerce")
    if "insat_lst_mean_K" in fused.columns:
        fused["insat_lst_mean_K"] = pd.to_numeric(fused["insat_lst_mean_K"], errors="coerce")
    if "insat_sst_mean_K" in fused.columns:
        fused["insat_sst_mean_K"] = pd.to_numeric(fused["insat_sst_mean_K"], errors="coerce")
    if "insat_rainfall_daily_mm" in fused.columns:
        fused["insat_rainfall_daily_mm"] = pd.to_numeric(fused["insat_rainfall_daily_mm"], errors="coerce")

    return fused.sort_values(["observation_date", "latitude_deg", "longitude_deg"]).reset_index(drop=True)


def build_matched_overlap_dataset(imd_frame: pd.DataFrame, insat_frame: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """Return only rows with a real IMD/INSAT overlap on date + latitude + longitude.

    This intentionally excludes all unmatched rows instead of fabricating values.
    """
    imd = harmonize_time(imd_frame, "observation_date").copy()
    imd = harmonize_spatial_grid(imd, "latitude_deg", "longitude_deg")
    imd = imd.groupby(["observation_date", "latitude_deg", "longitude_deg"], as_index=False).agg({
        "imd_rainfall_mm": "mean",
        "imd_max_temperature_C": "mean",
        "imd_min_temperature_C": "mean",
    })

    insat = aggregate_insat_daily(insat_frame)
    matched = imd.merge(
        insat,
        on=["observation_date", "latitude_deg", "longitude_deg"],
        how="inner",
    )
    matched = matched.sort_values(["observation_date", "latitude_deg", "longitude_deg"]).reset_index(drop=True)

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(out_path, index=False)

    return matched


def validate_fused_dataset(fused: pd.DataFrame) -> dict[str, Any]:
    """Compute a validation summary for the fused daily dataset."""
    summary = {
        "total_rows": int(len(fused)),
        "date_range": [
            fused["observation_date"].min().strftime("%Y-%m-%d") if "observation_date" in fused.columns else None,
            fused["observation_date"].max().strftime("%Y-%m-%d") if "observation_date" in fused.columns else None,
        ],
        "latitude_range": [
            float(fused["latitude_deg"].min()) if "latitude_deg" in fused.columns else None,
            float(fused["latitude_deg"].max()) if "latitude_deg" in fused.columns else None,
        ],
        "longitude_range": [
            float(fused["longitude_deg"].min()) if "longitude_deg" in fused.columns else None,
            float(fused["longitude_deg"].max()) if "longitude_deg" in fused.columns else None,
        ],
        "unique_dates": int(fused["observation_date"].nunique()) if "observation_date" in fused.columns else 0,
        "unique_grid_cells": int(fused[["latitude_deg", "longitude_deg"]].drop_duplicates().shape[0]) if {"latitude_deg", "longitude_deg"}.issubset(fused.columns) else 0,
        "duplicate_key_rows": int(fused.duplicated(subset=["observation_date", "latitude_deg", "longitude_deg"], keep=False).sum()),
        "matched_insat_rows": int(fused["insat_lst_mean_K"].notna().sum()) if "insat_lst_mean_K" in fused.columns else 0,
        "unmatched_imd_rows": int(fused["insat_lst_mean_K"].isna().sum()) if "insat_lst_mean_K" in fused.columns else 0,
    }

    for column in [
        "imd_rainfall_mm",
        "imd_max_temperature_C",
        "imd_min_temperature_C",
        "insat_lst_mean_K",
        "insat_lst_min_K",
        "insat_lst_max_K",
        "insat_rainfall_daily_mm",
        "insat_sst_mean_K",
    ]:
        if column in fused.columns:
            series = pd.to_numeric(fused[column], errors="coerce")
            summary[column] = {
                "valid": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100.0) if len(series) else 0.0,
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
            }

    return summary


def create_features(fused: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features from the fused daily dataset."""
    frame = fused.copy().sort_values(["latitude_deg", "longitude_deg", "observation_date"]).reset_index(drop=True)
    if "observation_date" not in frame.columns:
        raise ValueError("Fused dataset must contain observation_date to create temporal features")

    grouped = frame.groupby(["latitude_deg", "longitude_deg"], group_keys=False)

    target_columns = [
        "imd_rainfall_mm",
        "imd_max_temperature_C",
        "imd_min_temperature_C",
        "insat_lst_mean_K",
        "insat_sst_mean_K",
    ]

    for column in target_columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_lag_1d"] = grouped[column].shift(1)
        frame[f"{column}_lag_3d"] = grouped[column].shift(3)
        frame[f"{column}_lag_7d"] = grouped[column].shift(7)
        frame[f"{column}_rolling_3d"] = grouped[column].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        frame[f"{column}_rolling_7d"] = grouped[column].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    frame["temperature_range_C"] = frame.get("imd_max_temperature_C", pd.Series(np.nan, index=frame.index)) - frame.get("imd_min_temperature_C", pd.Series(np.nan, index=frame.index))
    frame["month"] = pd.to_datetime(frame["observation_date"]).dt.month
    frame["day_of_year"] = pd.to_datetime(frame["observation_date"]).dt.dayofyear

    return frame.reset_index(drop=True)


def build_pipeline() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load actual IMD and INSAT data, clean them, fuse them, validate, and create features."""
    imd_clean = clean_imd_data(load_imd_data())
    insat_clean = clean_insat_data(load_insat_data())
    fused = merge_imd_insat(imd_clean, insat_clean)
    matched_only = build_matched_overlap_dataset(imd_clean, insat_clean, FUSED_DIR / "imd_insat_fused_matched_only.csv")

    validation = validate_fused_dataset(fused)
    features = create_features(fused)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "imd_processed").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "insat_processed").mkdir(parents=True, exist_ok=True)
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    imd_clean.to_csv(PROCESSED_DIR / "imd_processed" / "imd_cleaned_daily.csv", index=False)
    insat_clean.to_csv(PROCESSED_DIR / "insat_processed" / "insat_cleaned_daily.csv", index=False)
    fused.to_csv(FUSED_DIR / "imd_insat_fused.csv", index=False)
    matched_only.to_csv(FUSED_DIR / "imd_insat_fused_matched_only.csv", index=False)
    features.to_csv(FEATURES_DIR / "climate_features.csv", index=False)

    report_path = PROCESSED_DIR / "dataset_report.json"
    report_path.write_text(json.dumps({
        "imd_files": [str(path.relative_to(PROJECT_ROOT)) for path in list_imd_files()],
        "insat_files": [str(path.relative_to(PROJECT_ROOT)) for path in list_insat_files()],
        "validation": validation,
        "strict_overlap_rows": len(matched_only),
        "strict_overlap_file": str((FUSED_DIR / "imd_insat_fused_matched_only.csv").relative_to(PROJECT_ROOT)),
    }, indent=2), encoding="utf-8")

    return fused, validation, features, matched_only


def main() -> None:
    fused, validation, features, matched_only = build_pipeline()
    print(f"Fused rows: {len(fused)}")
    print(f"Matched INSAT rows: {validation['matched_insat_rows']}")
    print(f"Unmatched IMD rows: {validation['unmatched_imd_rows']}")
    print(f"Strict matched-only rows: {len(matched_only)}")
    print(f"Fused output: {FUSED_DIR / 'imd_insat_fused.csv'}")
    print(f"Matched-only output: {FUSED_DIR / 'imd_insat_fused_matched_only.csv'}")
    print(f"Features output: {FEATURES_DIR / 'climate_features.csv'}")


if __name__ == "__main__":
    main()
