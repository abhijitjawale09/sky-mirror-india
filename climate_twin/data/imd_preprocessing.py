from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AxisSpec:
    """Describe one regular grid axis from a GrADS ctl file."""

    size: int
    start: float
    step: float

    def values(self) -> np.ndarray:
        """Return the coordinate values for the full axis."""

        return self.start + np.arange(self.size, dtype=float) * self.step


@dataclass(frozen=True)
class CtlMetadata:
    """Parsed metadata extracted from a GrADS ctl file."""

    ctl_path: Path
    grd_path: Path
    nx: int
    ny: int
    nt: int
    start_date: date
    time_step: str
    missing_value: float
    x_axis: AxisSpec
    y_axis: AxisSpec


@dataclass(frozen=True)
class SourcePair:
    """A paired ctl/grd file plus the detected product family."""

    kind: str
    year: int
    ctl_path: Path
    grd_path: Path


SUPPORTED_KINDS = {"rainfall", "tmax", "tmin"}
KIND_PRIORITY = ("rainfall", "tmax", "tmin")
TIME_STEP_ALIASES = {
    "dy": "D",
    "day": "D",
    "days": "D",
    "hr": "H",
    "hour": "H",
    "hours": "H",
    "mn": "T",
    "min": "T",
    "mins": "T",
    "mo": "MS",
    "mon": "MS",
    "month": "MS",
    "months": "MS",
    "yr": "YS",
    "year": "YS",
    "years": "YS",
}


def parse_ctl_file(ctl_path: Path) -> CtlMetadata:
    """Parse the ctl file and resolve its paired grd file.

    The parser extracts the grid dimensions, the start date, the time-step token,
    and the missing value code directly from the ctl contents.
    """

    text = ctl_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    x_match = _find_axis_match(lines, "xdef")
    y_match = _find_axis_match(lines, "ydef")
    t_match = _find_tdef_match(lines)
    missing_value = _find_missing_value(lines)
    grd_path = resolve_grd_path(ctl_path, text)

    x_axis = AxisSpec(size=x_match[0], start=x_match[1], step=x_match[2])
    y_axis = AxisSpec(size=y_match[0], start=y_match[1], step=y_match[2])
    start_date = _parse_ctl_date(t_match[1])

    return CtlMetadata(
        ctl_path=ctl_path,
        grd_path=grd_path,
        nx=x_axis.size,
        ny=y_axis.size,
        nt=t_match[0],
        start_date=start_date,
        time_step=t_match[2],
        missing_value=missing_value,
        x_axis=x_axis,
        y_axis=y_axis,
    )


def resolve_grd_path(ctl_path: Path, ctl_text: str) -> Path:
    """Resolve the matching grd file for a ctl file."""

    same_stem = ctl_path.with_suffix(".grd")
    if same_stem.exists():
        return same_stem

    uppercase_stem = ctl_path.with_suffix(".GRD")
    if uppercase_stem.exists():
        return uppercase_stem

    dset_match = re.search(r"^\s*dset\s+(.+)$", ctl_text, flags=re.IGNORECASE | re.MULTILINE)
    if dset_match:
        candidate = dset_match.group(1).strip().lstrip("^")
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = ctl_path.parent / candidate_path
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Could not resolve matching .grd file for {ctl_path}")


def infer_kind(path: Path, ctl_text: str) -> str:
    """Infer whether a file is rainfall, tmax, or tmin data."""

    haystack = f"{path.name}\n{ctl_text}".lower()
    if re.search(r"rain", haystack):
        return "rainfall"
    if re.search(r"tmax|maxtemp|max_t|max", haystack):
        return "tmax"
    if re.search(r"tmin|mintemp|min_t|min", haystack):
        return "tmin"
    raise ValueError(f"Unable to infer IMD product family from {path}")


def discover_source_pairs(input_dir: Path) -> list[SourcePair]:
    """Find all ctl/grd pairs beneath an input directory."""

    pairs: list[SourcePair] = []
    for ctl_path in sorted(input_dir.rglob("*.ctl")):
        try:
            ctl_text = ctl_path.read_text(encoding="utf-8", errors="ignore")
            metadata = parse_ctl_file(ctl_path)
            kind = infer_kind(ctl_path, ctl_text)
        except Exception as exc:  # pragma: no cover - logging path
            LOGGER.warning("Skipping %s: %s", ctl_path, exc)
            continue

        pairs.append(SourcePair(kind=kind, year=metadata.start_date.year, ctl_path=ctl_path, grd_path=metadata.grd_path))

    return pairs


def read_grd_cube(grd_path: Path, metadata: CtlMetadata) -> np.ndarray:
    """Read the binary grd file into a (time, latitude, longitude) cube."""

    expected_values = metadata.nt * metadata.ny * metadata.nx
    cube = np.fromfile(grd_path, dtype="<f4", count=expected_values)
    if cube.size != expected_values:
        raise ValueError(f"{grd_path.name} has {cube.size} values, expected {expected_values}")
    return cube.reshape((metadata.nt, metadata.ny, metadata.nx))


def cube_to_long_frame(cube: np.ndarray, metadata: CtlMetadata, value_column: str) -> pd.DataFrame:
    """Convert a GRD cube to a long-format dataframe."""

    dates = pd.date_range(start=metadata.start_date, periods=metadata.nt, freq=ctl_time_step_to_pandas_freq(metadata.time_step))
    latitudes = metadata.y_axis.values()
    longitudes = metadata.x_axis.values()

    flattened = cube.reshape(-1)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.values, metadata.ny * metadata.nx),
            "latitude": np.tile(np.repeat(latitudes, metadata.nx), metadata.nt),
            "longitude": np.tile(np.tile(longitudes, metadata.ny), metadata.nt),
            value_column: flattened,
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def clean_missing_values(
    frame: pd.DataFrame,
    value_column: str,
    missing_value: float,
    gap_limit: int = 3,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace ctl missing codes with NaN, interpolate short gaps, and drop the rest.

    I choose limited interpolation because these IMD products are daily grids and
    dropping every isolated missing cell would create unnecessary holes in the fused
    training table. The interpolation is constrained to short runs only; anything
    longer remains NaN and is dropped.
    """

    cleaned = frame.copy()
    before = len(cleaned)
    cleaned[value_column] = cleaned[value_column].replace(missing_value, np.nan)

    if gap_limit > 0:
        cleaned = cleaned.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)
        cleaned[value_column] = (
            cleaned.groupby(["latitude", "longitude"], sort=False)[value_column]
            .transform(lambda series: series.interpolate(limit=gap_limit, limit_direction="both"))
        )

    after_interpolation = int(cleaned[value_column].notna().sum())
    cleaned = cleaned.dropna(subset=[value_column]).reset_index(drop=True)

    return cleaned, {
        "rows_before": before,
        "rows_after_interpolation": after_interpolation,
        "rows_after_dropna": len(cleaned),
        "rows_dropped": before - len(cleaned),
    }


def ctl_time_step_to_pandas_freq(time_step: str) -> str:
    """Translate a ctl time-step token into a pandas date_range frequency."""

    match = re.fullmatch(r"(?P<count>\d+)?(?P<unit>[a-zA-Z]+)", time_step.strip())
    if not match:
        raise ValueError(f"Unrecognized ctl time-step token: {time_step}")

    count = match.group("count") or "1"
    unit = match.group("unit").lower()
    pandas_unit = TIME_STEP_ALIASES.get(unit)
    if pandas_unit is None:
        raise ValueError(f"Unsupported ctl time-step unit: {time_step}")
    return f"{count}{pandas_unit}"


def _find_axis_match(lines: list[str], axis_name: str) -> tuple[int, float, float]:
    pattern = re.compile(rf"^{axis_name}\s+(?P<size>\d+)\s+linear\s+(?P<start>[-+]?\d*\.?\d+)\s+(?P<step>[-+]?\d*\.?\d+)", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match:
            return int(match.group("size")), float(match.group("start")), float(match.group("step"))
    raise ValueError(f"Could not parse {axis_name} from ctl file")


def _find_tdef_match(lines: list[str]) -> tuple[int, str, str]:
    pattern = re.compile(r"^tdef\s+(?P<count>\d+)\s+linear\s+(?P<start>\S+)\s+(?P<step>\S+)", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match:
            return int(match.group("count")), match.group("start"), match.group("step")
    raise ValueError("Could not parse tdef from ctl file")


def _find_missing_value(lines: list[str]) -> float:
    patterns = (
        re.compile(r"^undef\s+(?P<value>[-+]?\d*\.?\d+)", re.IGNORECASE),
        re.compile(r"^missing\s+(?P<value>[-+]?\d*\.?\d+)", re.IGNORECASE),
    )
    for line in lines:
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                return float(match.group("value"))
    return -999.0


def _parse_ctl_date(token: str) -> date:
    token = token.strip().lower()
    for fmt in ("%d%b%Y", "%d%b%y"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse ctl start date token: {token}")


def _build_grid_frame(metadata: CtlMetadata) -> pd.DataFrame:
    dates = pd.date_range(start=metadata.start_date, periods=metadata.nt, freq=ctl_time_step_to_pandas_freq(metadata.time_step))
    latitudes = metadata.y_axis.values()
    longitudes = metadata.x_axis.values()
    return pd.DataFrame(
        {
            "date": np.repeat(dates.values, metadata.ny * metadata.nx),
            "latitude": np.tile(np.repeat(latitudes, metadata.nx), metadata.nt),
            "longitude": np.tile(np.tile(longitudes, metadata.ny), metadata.nt),
        }
    )


def regrid_to_target(
    source_frame: pd.DataFrame,
    source_metadata: CtlMetadata,
    target_metadata: CtlMetadata,
    value_column: str,
) -> pd.DataFrame:
    """Nearest-neighbor regrid a source frame onto the target grid.

    The target grid controls the output coordinate system.
    """

    target_grid = _build_grid_frame(target_metadata)
    target_grid["source_latitude"] = target_grid["latitude"].map(
        lambda value: _nearest_axis_value(value, source_metadata.y_axis)
    )
    target_grid["source_longitude"] = target_grid["longitude"].map(
        lambda value: _nearest_axis_value(value, source_metadata.x_axis)
    )

    regridded = target_grid.merge(
        source_frame[["date", "latitude", "longitude", value_column]],
        left_on=["date", "source_latitude", "source_longitude"],
        right_on=["date", "latitude", "longitude"],
        how="left",
        suffixes=("", "_source"),
    )

    regridded = regridded[["date", "latitude", "longitude", value_column]]
    return regridded.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)


def merge_sources(
    rainfall: pd.DataFrame,
    tmax: pd.DataFrame,
    tmin: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse rainfall, maximum temperature, and minimum temperature frames."""

    merged = rainfall.merge(tmax, on=["date", "latitude", "longitude"], how="left", suffixes=("", "_tmax"))
    merged = merged.merge(tmin, on=["date", "latitude", "longitude"], how="left", suffixes=("", "_tmin"))
    merged["year"] = pd.to_datetime(merged["date"]).dt.year.astype(int)
    return merged.sort_values(["year", "date", "latitude", "longitude"]).reset_index(drop=True)


def write_partitioned_parquet(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one parquet file per year partition."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Writing parquet partitions requires pyarrow. Install pyarrow and rerun the preprocessing script."
        ) from exc

    for year, year_frame in frame.groupby("year", sort=True):
        partition_dir = output_dir / f"year={int(year)}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / "part-000.parquet"
        year_frame.to_parquet(output_path, index=False)
        written.append(output_path)

    return written


def process_year(
    rainfall_pair: SourcePair,
    tmax_pair: SourcePair,
    tmin_pair: SourcePair,
    regrid_direction: str,
    gap_limit: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a single-year fused dataframe from the three IMD products."""

    rainfall_meta = parse_ctl_file(rainfall_pair.ctl_path)
    tmax_meta = parse_ctl_file(tmax_pair.ctl_path)
    tmin_meta = parse_ctl_file(tmin_pair.ctl_path)

    rainfall = cube_to_long_frame(read_grd_cube(rainfall_meta.grd_path, rainfall_meta), rainfall_meta, "rainfall_mm")
    tmax = cube_to_long_frame(read_grd_cube(tmax_meta.grd_path, tmax_meta), tmax_meta, "tmax_c")
    tmin = cube_to_long_frame(read_grd_cube(tmin_meta.grd_path, tmin_meta), tmin_meta, "tmin_c")

    rainfall, rainfall_stats = clean_missing_values(rainfall, "rainfall_mm", rainfall_meta.missing_value, gap_limit=gap_limit)
    tmax, tmax_stats = clean_missing_values(tmax, "tmax_c", tmax_meta.missing_value, gap_limit=gap_limit)
    tmin, tmin_stats = clean_missing_values(tmin, "tmin_c", tmin_meta.missing_value, gap_limit=gap_limit)

    if regrid_direction == "rain_to_temp":
        rainfall = regrid_to_target(rainfall, rainfall_meta, tmax_meta, "rainfall_mm")
        tmax = regrid_to_target(tmax, tmax_meta, tmax_meta, "tmax_c")
        tmin = regrid_to_target(tmin, tmin_meta, tmax_meta, "tmin_c")
    elif regrid_direction == "temp_to_rain":
        tmax = regrid_to_target(tmax, tmax_meta, rainfall_meta, "tmax_c")
        tmin = regrid_to_target(tmin, tmin_meta, rainfall_meta, "tmin_c")
        rainfall = regrid_to_target(rainfall, rainfall_meta, rainfall_meta, "rainfall_mm")
    else:
        raise ValueError("regrid_direction must be 'rain_to_temp' or 'temp_to_rain'")

    merged_tmax = rainfall.merge(tmax, on=["date", "latitude", "longitude"], how="left")
    before_tmax_drop = len(merged_tmax)
    tmax_missing = int(merged_tmax["tmax_c"].isna().sum())
    merged_tmax = merged_tmax.dropna(subset=["rainfall_mm", "tmax_c"]).reset_index(drop=True)
    after_tmax_drop = len(merged_tmax)

    merged_tmin = merged_tmax.merge(tmin, on=["date", "latitude", "longitude"], how="left")
    before_tmin_drop = len(merged_tmin)
    tmin_missing = int(merged_tmin["tmin_c"].isna().sum())
    merged = merged_tmin.dropna(subset=["rainfall_mm", "tmax_c", "tmin_c"]).reset_index(drop=True)
    after_tmin_drop = len(merged)

    stats = {
        "rainfall_rows_before_clean": rainfall_stats["rows_before"],
        "rainfall_rows_after_clean": rainfall_stats["rows_after_dropna"],
        "tmax_rows_before_clean": tmax_stats["rows_before"],
        "tmax_rows_after_clean": tmax_stats["rows_after_dropna"],
        "tmin_rows_before_clean": tmin_stats["rows_before"],
        "tmin_rows_after_clean": tmin_stats["rows_after_dropna"],
        "rows_before_tmax_drop": before_tmax_drop,
        "rows_after_tmax_drop": after_tmax_drop,
        "rows_missing_tmax_after_merge": tmax_missing,
        "rows_before_tmin_drop": before_tmin_drop,
        "rows_after_tmin_drop": after_tmin_drop,
        "rows_missing_tmin_after_merge": tmin_missing,
        "rows_after_final_dropna": len(merged),
        "rows_dropped_after_tmax": before_tmax_drop - after_tmax_drop,
        "rows_dropped_after_tmin": before_tmin_drop - after_tmin_drop,
        "rows_dropped_final": before_tmax_drop - len(merged),
    }
    return merged, stats


def process_folder(
    input_dir: Path,
    output_dir: Path,
    regrid_direction: str,
    gap_limit: int = 3,
) -> list[Path]:
    """Process all ctl/grd pairs in a folder tree and write year-partitioned parquet."""

    source_pairs = discover_source_pairs(input_dir)
    if not source_pairs:
        raise FileNotFoundError(f"No ctl files were found under {input_dir}")

    grouped: dict[int, dict[str, SourcePair]] = {}
    for pair in source_pairs:
        year_bucket = grouped.setdefault(pair.year, {})
        if pair.kind in year_bucket:
            LOGGER.warning(
                "Duplicate %s source for %s; keeping %s and ignoring %s",
                pair.kind,
                pair.year,
                year_bucket[pair.kind].grd_path.name,
                pair.grd_path.name,
            )
            continue
        year_bucket[pair.kind] = pair

    written_paths: list[Path] = []
    for year in sorted(grouped):
        year_pairs = grouped[year]
        missing = [kind for kind in SUPPORTED_KINDS if kind not in year_pairs]
        if missing:
            LOGGER.warning("Skipping %s because these products are missing: %s", year, ", ".join(sorted(missing)))
            continue

        LOGGER.info(
            "Processing %s with rainfall=%s, tmax=%s, tmin=%s",
            year,
            year_pairs["rainfall"].grd_path.name,
            year_pairs["tmax"].grd_path.name,
            year_pairs["tmin"].grd_path.name,
        )
        fused, stats = process_year(
            year_pairs["rainfall"],
            year_pairs["tmax"],
            year_pairs["tmin"],
            regrid_direction=regrid_direction,
            gap_limit=gap_limit,
        )
        LOGGER.info(
            "Year %s rows: rainfall %s->%s, tmax %s->%s, tmin %s->%s, final merge %s->%s",
            year,
            stats["rainfall_rows_before_clean"],
            stats["rainfall_rows_after_clean"],
            stats["tmax_rows_before_clean"],
            stats["tmax_rows_after_clean"],
            stats["tmin_rows_before_clean"],
            stats["tmin_rows_after_clean"],
            stats["rows_before_tmax_drop"],
            stats["rows_after_final_dropna"],
        )
        LOGGER.info(
            "Year %s merge drops: tmax missing=%s, tmax dropped=%s, tmin missing=%s, tmin dropped=%s, final dropped=%s",
            year,
            stats["rows_missing_tmax_after_merge"],
            stats["rows_dropped_after_tmax"],
            stats["rows_missing_tmin_after_merge"],
            stats["rows_dropped_after_tmin"],
            stats["rows_dropped_final"],
        )
        LOGGER.info("Year %s rows dropped at final merge cleanup: %s", year, stats["rows_dropped_final"])

        fused["year"] = fused["year"].astype(int)
        written_paths.extend(write_partitioned_parquet(fused, output_dir))

    return written_paths


def _nearest_axis_value(value: float, axis: AxisSpec) -> float:
    """Snap a value to the nearest coordinate on a regular grid axis."""

    index = int(round((value - axis.start) / axis.step))
    index = max(0, min(axis.size - 1, index))
    return float(axis.start + index * axis.step)
