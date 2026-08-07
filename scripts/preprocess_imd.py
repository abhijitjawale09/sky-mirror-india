#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.data.imd_preprocessing import process_folder


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the IMD preprocessing job."""

    parser = argparse.ArgumentParser(description="Preprocess IMD GRD/CTL climate data into year-partitioned Parquet.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing raw .ctl and .grd files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination folder for partitioned parquet output.")
    parser.add_argument(
        "--regrid-direction",
        required=True,
        choices=("rain_to_temp", "temp_to_rain"),
        help="Choose whether rainfall is snapped to the 1° temperature grid or temperature is snapped to the rainfall grid.",
    )
    parser.add_argument(
        "--gap-limit",
        type=int,
        default=3,
        help="Maximum consecutive missing daily values to interpolate per grid cell before dropping the remainder.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the folder-level preprocessing pipeline."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    written_paths = process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        regrid_direction=args.regrid_direction,
        gap_limit=args.gap_limit,
    )
    logging.info("Wrote %s parquet partitions", len(written_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())