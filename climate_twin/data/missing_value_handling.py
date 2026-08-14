from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = PROJECT_ROOT / "data" / "fused" / "imd_insat_fused.csv"
OUTPUT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "climate_dataset_missing_handled.csv"
OUTPUT_REPORT = PROJECT_ROOT / "data" / "processed" / "missing_value_report.json"


def inspect_missing_values(frame: pd.DataFrame) -> dict[str, Any]:
    """Return missing-value counts and percentages for each variable."""
    report: dict[str, Any] = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column != "observation_date" else frame[column]
        missing = int(series.isna().sum()) if hasattr(series, "isna") else 0
        total = int(len(frame))
        report[column] = {
            "missing_before": missing,
            "missing_pct_before": round((missing / total) * 100.0, 2) if total else 0.0,
            "valid_before": total - missing,
        }
    return report


def time_interpolate_short_gaps(frame: pd.DataFrame, value_column: str, max_gap: int = 3) -> pd.DataFrame:
    """Interpolate short temporal gaps within the same latitude/longitude cell.

    This method is intentionally conservative: only short gaps are filled, and only when
    the series is ordered by time and grouped by the same grid cell. No synthetic values are
    created for long gaps or for columns that are entirely missing.
    """
    result = frame.copy()
    if value_column not in result.columns:
        return result

    result = result.sort_values(["latitude_deg", "longitude_deg", "observation_date"]).copy()
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")

    def interpolate_group(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("observation_date")
        original = group[value_column].copy()
        group_index = pd.DatetimeIndex(group["observation_date"])
        interpolated = group.set_index("observation_date")[value_column].interpolate(
            method="time",
            limit=max_gap,
            limit_direction="both",
        )
        interpolated = interpolated.reindex(group["observation_date"])
        return interpolated.reset_index(drop=True)

    result[value_column] = (
        result.groupby(["latitude_deg", "longitude_deg"], group_keys=False)
        .apply(interpolate_group)
        .reset_index(level=0, drop=True)
    )

    return result


def preserve_rainfall_and_sst(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rainfall and SST missing unless actual source metadata confirms otherwise.

    The current project does not provide valid ocean-mask metadata or confirmed zero-rainfall
    flags for the missing values. Accordingly, missing rainfall/SST remain missing and are not
    backfilled or fabricated.
    """
    result = frame.copy()
    for column in ["imd_rainfall_mm", "insat_rainfall_daily_mm", "insat_sst_mean_K"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def handle_missing_values(source_path: str | Path = SOURCE_DATASET) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a conservative missing-data policy to the current climate dataset."""
    source_file = Path(source_path)
    if not source_file.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_file}")

    frame = pd.read_csv(source_file)
    before = inspect_missing_values(frame)

    processed = frame.copy()
    processed["observation_date"] = pd.to_datetime(processed["observation_date"], errors="coerce")
    processed["latitude_deg"] = pd.to_numeric(processed["latitude_deg"], errors="coerce")
    processed["longitude_deg"] = pd.to_numeric(processed["longitude_deg"], errors="coerce")
    for column in ["imd_rainfall_mm", "imd_max_temperature_C", "imd_min_temperature_C", "insat_lst_mean_K", "insat_lst_min_K", "insat_lst_max_K", "insat_rainfall_daily_mm", "insat_sst_mean_K"]:
        if column in processed.columns:
            processed[column] = pd.to_numeric(processed[column], errors="coerce")

    temp_columns = ["imd_min_temperature_C", "imd_max_temperature_C", "insat_lst_mean_K", "insat_lst_min_K", "insat_lst_max_K"]
    for column in temp_columns:
        if column in processed.columns:
            before_missing = int(processed[column].isna().sum())
            if before_missing > 0 and before_missing < len(processed):
                processed = time_interpolate_short_gaps(processed, column, max_gap=3)
            elif before_missing == 0:
                processed[column] = processed[column]

    processed = preserve_rainfall_and_sst(processed)

    after = inspect_missing_values(processed)
    report = {
        "source_file": str(source_file.relative_to(PROJECT_ROOT)),
        "rows_before": int(len(frame)),
        "rows_after": int(len(processed)),
        "missing_before": before,
        "missing_after": after,
        "method_used": {
            "temperature": "time-based interpolation within the same latitude/longitude grid cell for short gaps only",
            "rainfall": "no blind mean replacement; missing rainfall values preserved unless source metadata confirms zero rainfall",
            "sst": "no SST values created over land; valid ocean pixels only; missing values preserved when no valid ocean pixel exists",
        },
        "filled_values": {
            column: int(before.get(column, {}).get("missing_before", 0) - after.get(column, {}).get("missing_before", 0))
            for column in after.keys()
            if column in before
        },
    }

    return processed, report


def main() -> None:
    processed, report = handle_missing_values(SOURCE_DATASET)
    OUTPUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_PROCESSED, index=False)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Processed file: {OUTPUT_PROCESSED}")
    print(f"Report file: {OUTPUT_REPORT}")
    print("Missing values handled conservatively without synthetic data creation.")


if __name__ == "__main__":
    main()
