"""Fase 2a: extrae los GeoTIFF CDL de los zips nacionales a data/interim/cdl/ (idempotente).

Uso: python scripts/02a_extract_cdl.py [años...]  (por defecto, todos los zips completos presentes)
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.grid import cdl_path  # noqa: E402
from agrors.io import ROOT, read_manifest  # noqa: E402

PAT = re.compile(r"(?P<year>\d{4})_(?P<res>10|30)m_cdls\.zip$")


def main() -> int:
    want = {int(a) for a in sys.argv[1:]}
    manifest = read_manifest()
    for z in sorted((ROOT / "data" / "raw" / "cdl").glob("*_cdls.zip")):
        m = PAT.search(z.name)
        year, res = int(m["year"]), int(m["res"])
        if want and year not in want:
            continue
        if z.relative_to(ROOT).as_posix() not in manifest:
            continue  # descarga no verificada todavía
        dest = cdl_path(year, res)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if len(tifs) != 1:
                raise RuntimeError(f"{z.name}: se esperaba 1 .tif, hay {tifs}")
            tmp = dest.with_suffix(".tif.part")
            with zf.open(tifs[0]) as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 24)
            tmp.replace(dest)
            print("extraído", z.name, "→", dest.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
