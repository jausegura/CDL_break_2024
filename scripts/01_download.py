"""Fase 1 (orden: piloto primero): descarga de CDL, CSB, condados y CropSight-US (idempotente).

Uso: python scripts/01_download.py [pilot|full] [--skip-10m]
  pilot: CDL de los años del piloto (30 m y 10 m), CSB de las tres ventanas, condados, CropSight-US.
  full:  además, CDL 30 m del resto de años 2008-2025.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, download, load_config  # noqa: E402

PUBLIC = "Public domain (USDA NASS)"


def jobs(scope: str) -> list[tuple[str, Path, str, str]]:
    cfg = load_config()
    src = cfg["sources"]
    raw = ROOT / "data" / "raw"
    pilot_years = list(cfg["pilot"]["years"])
    out = [
        (
            src["counties"],
            raw / "boundaries" / Path(src["counties"]).name,
            "Public domain (US Census Bureau)",
            "Cartographic boundary counties 2023 1:500k",
        )
    ]
    out.append(
        (
            src["cropsight"],
            raw / "cropsight" / Path(src["cropsight"]).name,
            "CC BY 4.0",
            "CropSight-US v1.0.1, Zenodo 10.5281/zenodo.19501943",
        )
    )
    for y in pilot_years:
        name = src["cdl_30m_pattern"].format(year=y)
        out.append((src["cdl_base"] + name, raw / "cdl" / name, PUBLIC, f"CDL {y} nacional 30 m"))
    for y in [] if "--skip-10m" in sys.argv else src["cdl_10m_years"]:
        name = src["cdl_10m_pattern"].format(year=y)
        out.append((src["cdl_base"] + name, raw / "cdl" / name, PUBLIC, f"CDL {y} nacional 10 m"))
    for w in cfg["csb"]["windows"]:
        name = src["csb_pattern"].format(window=w)
        out.append((src["csb_base"] + name, raw / "csb" / name, PUBLIC, f"CSB {w} rev23"))
    if scope == "full":
        y0, y1 = cfg["full"]["years"]
        for y in sorted(set(range(y0, y1 + 1)) - set(pilot_years), reverse=True):
            name = src["cdl_30m_pattern"].format(year=y)
            out.append((src["cdl_base"] + name, raw / "cdl" / name, PUBLIC, f"CDL {y} nacional 30 m"))
    return out


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    logging.basicConfig(
        filename=ROOT / "results" / "logs" / "01_download.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    todo = jobs(scope)
    # Descargas en paralelo moderado; el MANIFEST se reescribe por fichero (escrituras breves).
    with ThreadPoolExecutor(max_workers=load_config()["compute"]["download_workers"]) as ex:
        futs = {ex.submit(download, url, dest, lic, notes): dest for url, dest, lic, notes in todo}
        for fut in as_completed(futs):
            try:
                logging.info("OK %s", fut.result())
                print("OK", futs[fut].name, flush=True)
            except Exception as e:  # noqa: BLE001
                logging.error("FALLO %s: %s", futs[fut], e)
                print("FALLO", futs[fut].name, e, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
