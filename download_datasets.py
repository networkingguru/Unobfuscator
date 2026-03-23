#!/usr/bin/env python3
"""Download missing DOJ datasets from archive.org and community mirrors.

Downloads datasets one at a time, extracts PDFs to pdf_cache/, and deletes
the zip immediately to conserve disk space. Tracks provenance for all
non-DOJ sources.

Usage:
    python download_datasets.py [--cache-dir ./pdf_cache] [--dry-run]
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from core.config import PDF_CACHE_DIR

console = Console()
logger = logging.getLogger(__name__)

# 10% disk reserve threshold
DISK_RESERVE_FRACTION = 0.10

# Retry configuration for transient download failures
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 10  # seconds; doubles each retry (10, 20, 40, 80, 160)

# Source types — used in provenance JSON and referenced by output_generator.
SOURCE_ARCHIVE_ORG = "archive.org"
SOURCE_COMMUNITY_GEEKEN = "community-mirror:geeken"

_ARCHIVE_ORG_LABEL = (
    "Source: Internet Archive mirror — DOJ original unavailable. "
    "No DOJ-published checksums exist; integrity cannot be verified against the original."
)
_COMMUNITY_MIRROR_LABEL = (
    "Source: Community mirror (GeekenDev) — DOJ original unavailable. "
    "No DOJ-published checksums exist; integrity cannot be verified against the original."
)

def _cache_dir_name(ds_id: int, release_batch: str | None = None) -> str:
    """Return the cache subdirectory name that matches the DB ``release_batch``.

    pdf_processor and text_recovery resolve cached PDFs via
    ``pdf_cache/{release_batch}/{filename}``.  Most datasets use the
    ``VOL{id:05d}`` convention, but Dataset 11 is stored as ``DataSet11``
    in the DB — pass *release_batch* to override.
    """
    return release_batch or f"VOL{ds_id:05d}"


DATASETS = [
    {
        "id": 3,
        "url": "https://archive.org/download/data-set-1/DataSet%203.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": "160231C8C689C76003976B609E55689530FC4832A1535CE13BFCD8F871C21E65",
    },
    {
        "id": 4,
        "url": "https://archive.org/download/data-set-1/DataSet%204.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": "E78948690C22B904D6B79AFE92A4EB1D7ABB8E746B0D025123826EBEE0DF8273",
    },
    {
        "id": 5,
        "url": "https://archive.org/download/data-set-1/DataSet%205.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": "EEDA87E747487D718E35B661D2D078FF08F0B0E80107C5F498FCE17AE4F298BA",
    },
    {
        "id": 6,
        "url": "https://archive.org/download/data-set-1/DataSet%206.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": None,
    },
    {
        "id": 7,
        "url": "https://archive.org/download/data-set-1/DataSet%207.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": None,
    },
    {
        "id": 8,
        "url": "https://doj-files.geeken.dev/doj_zips/original_archives/DataSet%208.zip",
        "source_type": SOURCE_COMMUNITY_GEEKEN,
        "source_label": _COMMUNITY_MIRROR_LABEL,
        "sha256": "558010B96B7980ED529A2AD4EA224123A59446927E4441D23A2E8E5C2361EE07",
    },
    {
        "id": 9,
        "url": "https://archive.org/download/data-set-9/DATA%20SET%209.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": None,
    },
    {
        "id": 12,
        "url": "https://archive.org/download/data-set-12_202601/DataSet%2012.zip",
        "source_type": SOURCE_ARCHIVE_ORG,
        "source_label": _ARCHIVE_ORG_LABEL,
        "sha256": None,
    },
]


def get_disk_usage(path: str):
    """Return shutil.disk_usage for the given path."""
    return shutil.disk_usage(path)


def check_disk_space(path: str, download_size: int) -> bool:
    """Check if downloading would leave at least 10% disk free.

    Returns True if safe to proceed, False if it would breach the reserve.
    """
    usage = get_disk_usage(path)
    free_after = usage.free - download_size
    return free_after >= (usage.total * DISK_RESERVE_FRACTION)


def get_remote_size(url: str) -> int:
    """Get the Content-Length of a remote file via HEAD request.

    Returns 0 if the server doesn't provide Content-Length.
    """
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        return int(resp.headers.get("content-length", 0))
    except Exception as e:
        logger.warning("Could not get remote size for %s: %s", url, e)
        return 0


def download_with_resume(url: str, dest_path: str) -> None:
    """Download a file with resume support using HTTP Range headers.

    If a partial .part file exists, attempts to resume from where it left off.
    On completion, renames .part to the final dest_path.
    """
    part_path = dest_path + ".part"
    existing_size = 0
    if os.path.exists(part_path):
        existing_size = os.path.getsize(part_path)

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        console.print(f"  [dim]Resuming from {existing_size / 1e6:.1f} MB...[/dim]")

    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                       timeout=httpx.Timeout(30, read=120)) as resp:
        if resp.status_code == 416:
            # Range not satisfiable — file already complete
            if os.path.exists(part_path):
                os.rename(part_path, dest_path)
            return

        if existing_size > 0 and resp.status_code != 206:
            # Server doesn't support Range — start over
            console.print("  [yellow]Server doesn't support resume — restarting download[/yellow]")
            existing_size = 0

        resp.raise_for_status()

        mode = "ab" if resp.status_code == 206 else "wb"
        total = int(resp.headers.get("content-length", 0))
        if resp.status_code == 206:
            total += existing_size
        downloaded = existing_size

        with open(part_path, mode) as f:
            chunks_since_flush = 0
            for chunk in resp.iter_bytes(chunk_size=1_048_576):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                chunks_since_flush += 1
                if chunks_since_flush >= 64:  # flush every ~64 MB
                    f.flush()
                    chunks_since_flush = 0
                if total > 0:
                    pct = downloaded / total * 100
                    console.print(
                        f"\r  [dim]{downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)[/dim]",
                        end="",
                    )
        console.print()  # newline after progress

    os.rename(part_path, dest_path)


def extract_and_cleanup(zip_path: str, extract_dir: str) -> int:
    """Extract a zip file to extract_dir and delete the zip.

    Returns the number of files extracted.
    """
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Flatten: extract all files directly into extract_dir
            filename = os.path.basename(member.filename)
            if not filename:
                continue
            target = os.path.join(extract_dir, filename)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    os.remove(zip_path)
    return count


def write_provenance(prov_path: str, dataset_id: int, entry: dict) -> None:
    """Write or update a provenance entry for a dataset."""
    data = load_provenance(prov_path)
    entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    data[str(dataset_id)] = entry
    with open(prov_path, "w") as f:
        json.dump(data, f, indent=2)


def load_provenance(prov_path: str) -> dict:
    """Load provenance data from JSON. Returns empty dict if file doesn't exist."""
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            return json.load(f)
    return {}


def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@click.command()
@click.option("--cache-dir", default=str(PDF_CACHE_DIR), show_default=True,
              help="Directory to extract PDFs into")
@click.option("--dry-run", is_flag=True, help="Show what would be downloaded without doing it")
def main(cache_dir: str, dry_run: bool):
    """Download missing DOJ datasets from archive.org and community mirrors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    prov_path = str(cache / "provenance.json")

    # Show plan
    table = Table(title="Dataset Download Plan")
    table.add_column("Dataset", style="bold")
    table.add_column("Source")
    table.add_column("Status")

    for ds in DATASETS:
        ds_dir = cache / _cache_dir_name(ds["id"], ds.get("release_batch"))
        if ds_dir.exists() and any(ds_dir.iterdir()):
            status = "[green]already downloaded[/green]"
        else:
            status = "[yellow]pending[/yellow]"
        table.add_row(f"Dataset {ds['id']}", ds["source_type"], status)

    console.print(table)

    if dry_run:
        console.print("\n[dim]Dry run — no downloads performed.[/dim]")
        return

    for ds in DATASETS:
        ds_id = ds["id"]
        ds_dir = cache / _cache_dir_name(ds["id"], ds.get("release_batch"))

        # Skip if already downloaded
        if ds_dir.exists() and any(ds_dir.iterdir()):
            console.print(f"\n[green]Dataset {ds_id}: already in {ds_dir}, skipping.[/green]")
            continue

        console.print(f"\n[bold]Dataset {ds_id}[/bold]: {ds['url']}")

        # Get remote file size
        remote_size = get_remote_size(ds["url"])
        if remote_size > 0:
            console.print(f"  Size: {_fmt_bytes(remote_size)}")
        else:
            console.print("  [yellow]Could not determine file size — proceeding cautiously[/yellow]")

        # Check disk space (use remote_size, or skip check if unknown)
        if remote_size > 0 and not check_disk_space(str(cache), remote_size):
            usage = get_disk_usage(str(cache))
            console.print(
                f"  [red]ABORTING: downloading {_fmt_bytes(remote_size)} would leave "
                f"less than 10% disk free ({_fmt_bytes(usage.free)} free of "
                f"{_fmt_bytes(usage.total)} total).[/red]"
            )
            console.print("[red]Stopping all downloads to preserve disk space.[/red]")
            sys.exit(1)

        # Download with retries (keeps .part file for resume across attempts)
        zip_name = f"DataSet_{ds_id}.zip"
        zip_path = str(cache / zip_name)
        downloaded = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                download_with_resume(ds["url"], zip_path)
                downloaded = True
                break
            except Exception as e:
                console.print(f"  [red]Download failed (attempt {attempt}/{MAX_RETRIES}): {e}[/red]")
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    console.print(f"  [dim]Retrying in {wait}s...[/dim]")
                    time.sleep(wait)
                else:
                    console.print(f"  [red]Giving up on Dataset {ds_id} after {MAX_RETRIES} attempts. "
                                  f".part file preserved for next run.[/red]")
        if not downloaded:
            continue

        # Verify SHA-256 if available
        sha256_mismatch = False
        sha = hashlib.sha256()
        with open(zip_path, "rb") as f:
            for block in iter(lambda: f.read(1_048_576), b""):
                sha.update(block)
        actual_sha = sha.hexdigest().upper()

        if ds.get("sha256"):
            expected = ds["sha256"].upper()
            if actual_sha != expected:
                sha256_mismatch = True
                console.print(f"  [yellow]SHA-256 MISMATCH — archive may have been re-uploaded.[/yellow]")
                console.print(f"  Expected: {expected}")
                console.print(f"  Got:      {actual_sha}")
                console.print(f"  [yellow]No DOJ-published checksums exist, so the original cannot be "
                              f"verified. Proceeding with extraction.[/yellow]")
            else:
                console.print(f"  [green]SHA-256 verified.[/green]")

        # Extract
        ds_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"  Extracting to {ds_dir}...")
        try:
            count = extract_and_cleanup(zip_path, str(ds_dir))
            console.print(f"  [green]Extracted {count} files.[/green]")
        except Exception as e:
            console.print(f"  [red]Extraction failed: {e}[/red]")
            # Clean up zip if it still exists
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

        # Write provenance — record mismatch so output PDFs carry the warning
        prov_label = ds["source_label"]
        if sha256_mismatch:
            prov_label += (" SHA-256 did not match expected hash at download time; "
                           "archive may have been re-uploaded.")

        write_provenance(prov_path, ds_id, {
            "source_url": ds["url"],
            "source_type": ds["source_type"],
            "source_label": prov_label,
            "sha256_expected": ds.get("sha256"),
            "sha256_actual": actual_sha,
            "sha256_match": not sha256_mismatch,
            "files_extracted": count,
        })

        # Post-extraction disk check
        usage = get_disk_usage(str(cache))
        free_pct = usage.free / usage.total * 100
        console.print(f"  Disk: {_fmt_bytes(usage.free)} free ({free_pct:.1f}%)")

    console.print("\n[bold green]Download complete.[/bold green]")


if __name__ == "__main__":
    main()
