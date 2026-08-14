#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.data.insat_merge import merge_insat_with_imd
from climate_twin.data.loader import load_training_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract INSAT L2B data and merge it with the IMD fused climate table.")
    parser.add_argument("--imd-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "climate_training_data.csv", help="Path to the base IMD fused CSV.")
    parser.add_argument("--insat-dir", type=Path, required=True, help="Directory that contains INSAT HDF5 files.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "climate_training_data_with_insat.csv", help="Destination CSV for the merged dataset.")
    parser.add_argument("--extract-only", action="store_true", help="Only extract the raw INSAT values to a CSV and do not merge with IMD.")
    parser.add_argument("--extract-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "insat_extracted.csv", help="Destination CSV for extracted INSAT-only values.")
    parser.add_argument("--value-name-hint", default="LST", help="Likely INSAT variable name to search for in HDF5 content.")
    parser.add_argument("--years-back", type=int, default=15, help="Only process files from the last N calendar years.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.extract_only:
        from climate_twin.data.insat_merge import extract_insat_to_csv

        extracted = extract_insat_to_csv(
            args.insat_dir,
            args.extract_output,
            value_name_hint=args.value_name_hint,
            years_back=args.years_back,
        )
        print(f"Extracted INSAT-only output saved to {args.extract_output}")
        print(f"Rows written: {len(extracted)}")
        return 0

    if not args.imd_csv.exists():
        raise FileNotFoundError(f"Base IMD CSV not found: {args.imd_csv}. Run the IMD training pipeline first.")

    frame, _ = load_training_observations()
    merged = merge_insat_with_imd(
        frame,
        args.insat_dir,
        output_path=args.output,
        value_name_hint=args.value_name_hint,
        years_back=args.years_back,
    )
    print(f"Merged IMD + INSAT output saved to {args.output}")
    print(f"Coverage: {merged['insat_lst_c'].notna().sum()} / {len(merged)} rows have INSAT values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
