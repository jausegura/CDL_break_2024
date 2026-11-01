"""Descargas idempotentes con reanudación, verificación sha256 y registro en data/MANIFEST.csv."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import threading
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "MANIFEST.csv"
_LOCK = threading.Lock()
FIELDS = ["file", "source_url", "download_date", "sha256", "license", "notes"]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))


def sha256sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def write_manifest(rows: dict[str, dict]) -> None:
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({k: rows[key].get(k, "") for k in FIELDS})


def remote_size(url: str, retries: int = 6) -> int | None:
    for attempt in range(retries):
        try:
            r = requests.head(url, allow_redirects=True, timeout=120)
            r.raise_for_status()
            n = r.headers.get("Content-Length")
            return int(n) if n else None
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2**attempt)
    return None


def download(url: str, dest: Path, license_: str, notes: str = "", retries: int = 8) -> Path:
    """Descarga `url` en `dest` reanudando si hay un parcial. No repite si el MANIFEST ya tiene
    el fichero y su sha256 coincide."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    rel = dest.relative_to(ROOT).as_posix()
    rows = read_manifest()
    if dest.exists() and rel in rows and rows[rel]["sha256"]:
        return dest
    size = remote_size(url)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                if have and r.status_code != 206:
                    have = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                with open(part, "ab") as f:
                    for block in r.iter_content(chunk_size=1 << 22):
                        f.write(block)
            if size is None:
                break
        except (requests.RequestException, OSError):
            time.sleep(min(600, 15 * 2**attempt))
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f"Descarga incompleta: {url}")
    part.replace(dest)
    digest = sha256sum(dest)
    with _LOCK:
        rows = read_manifest()
        rows[rel] = {
            "file": rel,
            "source_url": url,
            "download_date": dt.date.today().isoformat(),
            "sha256": digest,
            "license": license_,
            "notes": notes,
        }
        write_manifest(rows)
    return dest
