"""Fase 1b: superficie sembrada/cosechada NASS Quick Stats por condado y estado (2008-2025).

Idempotente: un CSV por (nivel, cultivo, año) en data/raw/quickstats/. La clave de la API no se
escribe en el MANIFEST ni en los logs.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.env import get  # noqa: E402
from agrors.io import _LOCK, ROOT, load_config, read_manifest, sha256sum, write_manifest  # noqa: E402

COMMODITIES = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "HAY"]  # HAY: solo superficie cosechada
STATS = ["AREA PLANTED", "AREA HARVESTED"]


def fetch(params: dict, retries: int = 6) -> str:
    cfg = load_config()
    for attempt in range(retries):
        try:
            r = requests.get(
                cfg["sources"]["quickstats_api"],
                params={**params, "key": get("NASS_API_KEY", True), "format": "CSV"},
                timeout=300,
            )
            if r.status_code == 400 and "exceeds limit" in r.text:
                raise RuntimeError("Consulta supera 50.000 filas")
            if r.status_code == 400 and "no data" in r.text.lower():
                return ""
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(10 * 2**attempt)
    raise RuntimeError(f"Quick Stats falla: {params}")


def main() -> int:
    cfg = load_config()
    y0, y1 = cfg["full"]["years"]
    out_dir = ROOT / "data" / "raw" / "quickstats"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Uso opcional: python scripts/01b_quickstats.py STATE HAY  (restringe nivel y cultivo)
    levels = [sys.argv[1]] if len(sys.argv) > 1 else ["COUNTY", "STATE"]
    comms = [sys.argv[2]] if len(sys.argv) > 2 else COMMODITIES
    for level in levels:
        for comm in comms:
            for year in range(y0, y1 + 1):
                dest = out_dir / f"qs_{level.lower()}_{comm.lower()}_{year}.csv"
                rel = dest.relative_to(ROOT).as_posix()
                if dest.exists() and rel in read_manifest():
                    continue
                params = {
                    "source_desc": "SURVEY",
                    "commodity_desc": comm,
                    "year": year,
                    "agg_level_desc": level,
                    "statisticcat_desc": STATS,
                }
                text = fetch(params)
                rows = list(csv.DictReader(io.StringIO(text))) if text else []
                dest.write_text(text, encoding="utf-8")
                with _LOCK:
                    m = read_manifest()
                    m[rel] = {
                        "file": rel,
                        "source_url": "https://quickstats.nass.usda.gov/api/api_GET/?source_desc=SURVEY"
                        f"&commodity_desc={comm}&year={year}&agg_level_desc={level}"
                        "&statisticcat_desc=AREA PLANTED|AREA HARVESTED",
                        "download_date": dt.date.today().isoformat(),
                        "sha256": sha256sum(dest),
                        "license": "Public domain (USDA NASS)",
                        "notes": f"{len(rows)} filas",
                    }
                    write_manifest(m)
                print(level, comm, year, len(rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
