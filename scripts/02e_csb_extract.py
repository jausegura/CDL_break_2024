"""Fase 2e: extrae los polígonos CSB de los estados del piloto a GeoParquet (EPSG:5070).

Salida: data/interim/csb/csb{ventana}_{estado}.parquet con CSBID, CSBACRES, CDL<años>, STATEFIPS,
CNTYFIPS, INSIDE_X/Y y geometría. Idempotente.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config, read_manifest  # noqa: E402

STATE_FIPS = {"IA": "19", "IL": "17", "NE": "31", "KS": "20", "PA": "42", "CA": "06"}


def gdb_path(window: str) -> str:
    z = ROOT / "data" / "raw" / "csb" / f"NationalCSB_{window}_rev23.zip"
    short = window[2:4] + window[7:9]
    with zipfile.ZipFile(z) as zf:
        root = zf.namelist()[0].split("/")[0]
    return f"/vsizip/{z.as_posix()}/{root}/CSB{short}.gdb"


def main() -> int:
    cfg = load_config()
    out_dir = ROOT / "data" / "interim" / "csb"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    for window in cfg["csb"]["windows"]:
        z = ROOT / "data" / "raw" / "csb" / f"NationalCSB_{window}_rev23.zip"
        if z.relative_to(ROOT).as_posix() not in manifest:
            print("sin descargar:", z.name)
            continue
        todo = [s for s in cfg["pilot"]["states"] if not (out_dir / f"csb{window}_{s}.parquet").exists()]
        if not todo:
            continue
        path = gdb_path(window)
        info = pyogrio.read_info(path)
        cdl_cols = [f for f in info["fields"] if f.startswith("CDL")]
        cols = ["CSBID", "CSBACRES", *cdl_cols, "STATEFIPS", "CNTYFIPS", "INSIDE_X", "INSIDE_Y"]
        fips = ",".join(f"'{STATE_FIPS[s]}'" for s in todo)
        gdf = pyogrio.read_dataframe(path, columns=cols, where=f"STATEFIPS IN ({fips})")
        for s in todo:
            sub = gdf[gdf.STATEFIPS == STATE_FIPS[s]].reset_index(drop=True)
            sub.to_parquet(out_dir / f"csb{window}_{s}.parquet", index=False)
            print(window, s, len(sub), "polígonos", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
