#!/usr/bin/env python3
"""
MOSDAC SFTP Bulk Downloader
----------------------------
Downloads all files from a MOSDAC data order via SFTP.

Setup (run these in your terminal BEFORE running this script):
    macOS/Linux:
        export MOSDAC_USER="your_mosdac_username"
        export MOSDAC_PASS="your_mosdac_password"
    Windows (PowerShell):
        $env:MOSDAC_USER="your_mosdac_username"
        $env:MOSDAC_PASS="your_mosdac_password"

Install dependency:
    pip install paramiko --break-system-packages   # (or just: pip install paramiko)

Usage:
    # First, always dry-run to confirm the remote path is correct:
    python mosdac_download.py --remote-path /orders/Aug2026_187914 --local-dir ./data/raw/insat --dry-run

    # Then run for real:
    python mosdac_download.py --remote-path /orders/Aug2026_187914 --local-dir ./data/raw/insat

Note: You must find the exact --remote-path yourself by logging into
https://mosdac.gov.in/sso-download or connecting once with FileZilla/WinSCP
to see your order's folder structure. It will NOT be exactly "/orders/<id>" --
that's just a placeholder guess.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Missing dependency. Run: pip install paramiko --break-system-packages")
    sys.exit(1)

HOST = "download.mosdac.gov.in"
PORT = 22
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


def connect() -> paramiko.SFTPClient:
    username = os.environ.get("MOSDAC_USER")
    password = os.environ.get("MOSDAC_PASS")
    if not username or not password:
        print("ERROR: Set MOSDAC_USER and MOSDAC_PASS environment variables first.")
        sys.exit(1)

    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=username, password=password)
    return paramiko.SFTPClient.from_transport(transport)


def list_remote_files(sftp: paramiko.SFTPClient, remote_path: str) -> list[str]:
    try:
        entries = sftp.listdir_attr(remote_path)
    except FileNotFoundError:
        print(f"ERROR: Remote path not found: {remote_path}")
        print("Double-check the exact folder path from your MOSDAC order page.")
        sys.exit(1)

    files = [entry.filename for entry in entries if not str(entry.longname).startswith("d")]
    if not files:
        print(f"WARNING: No files found in {remote_path}. Path may be wrong or order expired.")
    return files


def download_with_retry(sftp: paramiko.SFTPClient, remote_file: str, local_file: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sftp.get(remote_file, str(local_file))
            return True
        except Exception as exc:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed for {remote_file.split('/')[-1]}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk download files from MOSDAC via SFTP.")
    parser.add_argument("--remote-path", required=True, help="Remote folder path on MOSDAC SFTP server")
    parser.add_argument("--local-dir", required=True, help="Local folder to save downloaded files")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {HOST}...")
    sftp = connect()
    print("Connected.")

    files = list_remote_files(sftp, args.remote_path)
    total = len(files)
    print(f"Found {total} file(s) in {args.remote_path}")

    if args.dry_run:
        for name in files:
            print(f"  [dry-run] {name}")
        sftp.close()
        return

    failed: list[str] = []
    downloaded_bytes = 0

    for index, filename in enumerate(files, start=1):
        remote_file = f"{args.remote_path.rstrip('/')}/{filename}"
        local_file = local_dir / filename

        success = download_with_retry(sftp, remote_file, local_file)
        if success:
            size = local_file.stat().st_size
            downloaded_bytes += size
            print(f"[{index}/{total}] OK  {filename} ({size / 1024:.1f} KB)")
        else:
            failed.append(filename)
            print(f"[{index}/{total}] FAIL {filename}")

    sftp.close()

    print("\n--- Summary ---")
    print(f"Downloaded: {total - len(failed)}/{total}")
    print(f"Total size: {downloaded_bytes / (1024 * 1024):.2f} MB")
    if failed:
        print(f"Failed files ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
