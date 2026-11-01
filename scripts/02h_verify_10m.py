"""Fase 2h: verifica el 10 m del almacén Icechunk y el remuestreo NASS 30 m contra el 10 m OFICIAL.

Requiere el zip oficial completo data/raw/cdl/{año}_10m_cdls.zip (se extrae el .tif a
data/interim/cdl/cdl_{año}_10m_official.tif). En bloques aleatorios del piloto con datos compara:
  (a) 10 m oficial vs 10 m del almacén (data/interim/cdl/cdl_{año}_10m.tif)
  (b) NASS 30 m oficial vs subpíxel central / mayoría 3x3 del 10 m oficial
Salida: results/tables/verify_10m.csv
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import center3, majority3  # noqa: E402
from agrors.grid import cdl_path, read_aligned, window10  # noqa: E402
from agrors.io import ROOT, load_config, read_manifest  # noqa: E402
from agrors.layers import Layers  # noqa: E402

N_BLOCKS = 60


def official(year: int) -> Path | None:
    z = ROOT / "data" / "raw" / "cdl" / f"{year}_10m_cdls.zip"
    if z.relative_to(ROOT).as_posix() not in read_manifest():
        return None
    out = ROOT / "data" / "interim" / "cdl" / f"cdl_{year}_10m_official.tif"
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
            tmp = out.with_suffix(".tif.part")
            with zf.open(name) as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 24)
            tmp.replace(out)
    return out


def main() -> int:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    L = Layers()
    counties = pd.read_csv(ROOT / "data" / "processed" / "counties.csv")
    pilot = counties.loc[counties.STUSPS.isin(cfg["pilot"]["states"]), "zone"].to_numpy()
    rows = []
    for year in cfg["sources"]["cdl_10m_years"]:
        off = official(year)
        if off is None:
            print(year, "sin zip oficial 10 m")
            continue
        with (
            rasterio.open(off) as so,
            rasterio.open(cdl_path(year, 10)) as ss,
            rasterio.open(cdl_path(year, 30)) as s30,
        ):
            got, tried = 0, 0
            while got < N_BLOCKS and tried < 5000:
                tried += 1
                r = int(rng.integers(0, L.shape30[0] - 512))
                c = int(rng.integers(0, L.shape30[1] - 512))
                w = Window(c, r, 512, 512)
                z = L.zone(w)
                if np.isin(z, pilot).mean() < 0.9:
                    continue
                a = read_aligned(so, L.t10, window10(w), (1536, 1536))
                b = read_aligned(ss, L.t10, window10(w), (1536, 1536))
                n30 = read_aligned(s30, L.t30, w, (512, 512))
                m = n30 > 0
                if m.mean() < 0.5:
                    continue
                got += 1
                rows.append(
                    {
                        "year": year,
                        "col30": c,
                        "row30": r,
                        "agree_official_vs_store": float((a == b).mean()),
                        "agree_nass30_vs_center": float((n30[m] == center3(a)[m]).mean()),
                        "agree_nass30_vs_majority": float((n30[m] == majority3(a)[m]).mean()),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "verify_10m.csv", index=False)
    if len(df):
        print(df.groupby("year").mean(numeric_only=True).round(5).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
