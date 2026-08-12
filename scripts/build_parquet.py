"""Convert processed CSV training table into partitioned Parquet for scalable ops.

Writes to `data/processed/parquet/` partitioned by `year` and `region`.
If `pyarrow` is not installed, falls back to writing per-year CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def write_parquet(df: pd.DataFrame, out_dir: Path):
    try:
        import pyarrow as pa  # noqa: F401
        engine = "pyarrow"
    except Exception:
        engine = None

    if engine:
        out_dir.mkdir(parents=True, exist_ok=True)
        # add partition cols
        df["year"] = pd.to_datetime(df["date"]).dt.year
        # write partitioned parquet
        df.to_parquet(out_dir / "climate_training.parquet", engine=engine, partition_cols=["year", "region"], index=False)
        print(f"Wrote partitioned Parquet to {out_dir}")
    else:
        # fallback: write per-year CSV files
        csv_dir = out_dir / "csv_by_year"
        csv_dir.mkdir(parents=True, exist_ok=True)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        for y, part in df.groupby("year"):
            path = csv_dir / f"climate_training_{y}.csv"
            part.to_csv(path, index=False)
        print(f"pyarrow not available — wrote per-year CSVs to {csv_dir}")


def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "climate_training_data.csv"
    out_dir = PROJECT_ROOT / "data" / "processed" / "parquet"
    if not data_path.exists():
        raise SystemExit(f"Missing input CSV: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    write_parquet(df, out_dir)


if __name__ == "__main__":
    main()
