#!/usr/bin/env python3
"""Download IMD datasets (rainfall, max temp, min temp) into `data/raw/`.

The script checks for existing files and skips downloads if they are present.
It attempts to find a direct archive or data link on the IMD landing page and
downloads the first file that looks like an archive or grid file. If no direct
asset is found the landing page HTML is saved for manual inspection.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "climate_twin" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_SOURCES = {
    "rainfall": "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
    "tmax": "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
    "tmin": "https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
}

ASSET_EXT_RE = re.compile(r"href=[\"']([^\"']+\.(?:zip|gz|tgz|tar|nc|txt|csv))(?:[\"'])", re.IGNORECASE)


def find_asset_link(html: str, base_url: str) -> str | None:
    m = ASSET_EXT_RE.search(html)
    if m:
        link = m.group(1)
        return urljoin(base_url, link)

    # fallback: look for any .zip or .nc in hrefs
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    for href in hrefs:
        if any(href.lower().endswith(ext) for ext in (".zip", ".nc", ".gz", ".tar", ".csv", ".txt")):
            return urljoin(base_url, href)

    return None


def download_url(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)


def fetch_source(name: str, landing_url: str, force: bool = False) -> None:
    # Determine a friendly filename prefix
    parsed = urlparse(landing_url)
    prefix = name

    # If any existing file matches the prefix, skip unless force
    existing = list(RAW_DIR.glob(f"{prefix}*"))
    if existing and not force:
        print(f"Skipping {name}: found existing file(s): {[p.name for p in existing]}")
        return

    resp = requests.get(landing_url, timeout=30)
    if resp.status_code != 200:
        print(f"Warning: unable to fetch landing page {landing_url} (status {resp.status_code})")
        return

    asset = find_asset_link(resp.text, landing_url)
    if asset:
        fname = Path(asset).name
        dest = RAW_DIR / fname
        try:
            download_url(asset, dest)
        except Exception as exc:
            print(f"Download failed for {asset}: {exc}")
    else:
        # Save landing page for manual inspection
        fname = f"{prefix}_landing.html"
        dest = RAW_DIR / fname
        print(f"No direct asset found for {name}; saving landing page to {dest}")
        dest.write_text(resp.text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IMD datasets into climate_twin/data/raw/")
    parser.add_argument("--force", action="store_true", help="Redownload even if files exist")
    parser.add_argument("--only", choices=list(DATA_SOURCES.keys()), help="Only fetch a single source")
    args = parser.parse_args()

    targets = {k: v for k, v in DATA_SOURCES.items() if (args.only is None or args.only == k)}

    for name, url in targets.items():
        try:
            fetch_source(name, url, force=args.force)
        except Exception as exc:
            print(f"Error fetching {name}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
