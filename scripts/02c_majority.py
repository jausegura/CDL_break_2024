"""Fase 2c: CDL 10 m → 30 m por mayoría 3x3 (variante maj30) y control del remuestreo NASS.

Para cada año de la era nueva escribe data/interim/cdl/cdl_{año}_maj30.tif en la rejilla de
condados y, de paso, mide por estado (dentro de CONUS):
  - acuerdo entre el producto NASS 30 m y el subpíxel central del 10 m (¿es vecino más próximo?)
  - acuerdo entre NASS 30 m y la mayoría 3x3
Salida: results/tables/resampling_check.csv
"""

from __future__ import annotations

import atexit
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import center3, majority3  # noqa: E402
from agrors.grid import bounded_map, cdl_path, county_path, maj30_path, read_aligned, window10, windows  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.layers import Layers  # noqa: E402

_L: Layers | None = None
_STATE_OF_ZONE: np.ndarray | None = None


def _init():
    global _L, _STATE_OF_ZONE
    _L = Layers()
    atexit.register(_L.close)
    c = pd.read_csv(ROOT / "data" / "processed" / "counties.csv", dtype={"STATEFP": str})
    lut = np.zeros(int(c.zone.max()) + 1, dtype=np.uint8)
    lut[c.zone.values] = c.STATEFP.astype(int).values
    _STATE_OF_ZONE = lut


def _block(args):
    year, win = args
    zone = _L.zone(win)
    shape = zone.shape
    a10 = read_aligned(_L._src(cdl_path(year, 10)), _L.t10, window10(win), (shape[0] * 3, shape[1] * 3))
    if not a10.any():
        return win, None, None
    maj = majority3(a10)
    cen = center3(a10)
    nass = read_aligned(_L._src(cdl_path(year, 30)), _L.t30, win, shape)
    m = (zone > 0) & (nass > 0) & (cen > 0)  # solo donde hay 10 m (el extracto puede ser del piloto)
    st = _STATE_OF_ZONE[zone[m]]
    stats = np.column_stack(
        [
            np.bincount(st, minlength=100),
            np.bincount(st, weights=(nass[m] == cen[m]), minlength=100),
            np.bincount(st, weights=(nass[m] == maj[m]), minlength=100),
        ]
    )
    return win, maj, stats


def main() -> int:
    cfg = load_config()
    n_workers = cfg["compute"]["n_workers"]
    rows = []
    for year in cfg["sources"]["cdl_10m_years"]:
        out = maj30_path(year)
        if out.exists() or not cdl_path(year, 10).exists():
            continue
        with rasterio.open(county_path()) as z:
            prof = z.profile.copy()
            wins = list(windows(z.width, z.height, cfg["compute"]["block_size"]))
        prof.update(dtype="uint8", nodata=0)
        tmp = out.with_suffix(".tif.part")
        total = np.zeros((100, 3))
        with (
            rasterio.open(tmp, "w", **prof) as dst,
            ProcessPoolExecutor(cfg["compute"]["n_workers"], initializer=_init) as ex,
        ):
            for win, maj, stats in bounded_map(ex, _block, [(year, w) for w in wins], 2 * n_workers):
                if maj is not None:
                    dst.write(maj, 1, window=win)
                    total += stats
        tmp.replace(out)
        for fips in np.nonzero(total[:, 0])[0]:
            n, eq_c, eq_m = total[fips]
            rows.append(
                {
                    "year": year,
                    "statefp": f"{fips:02d}",
                    "n_pixels30": int(n),
                    "agree_nass_center": eq_c / n,
                    "agree_nass_majority": eq_m / n,
                }
            )
        n, eq_c, eq_m = total.sum(0)
        rows.append(
            {
                "year": year,
                "statefp": "CONUS",
                "n_pixels30": int(n),
                "agree_nass_center": eq_c / n,
                "agree_nass_majority": eq_m / n,
            }
        )
        print(year, "acuerdo NASS-central", eq_c / n, "NASS-mayoría", eq_m / n, flush=True)
    if rows:
        path = ROOT / "results" / "tables" / "resampling_check.csv"
        old = pd.read_csv(path, dtype={"statefp": str}) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["year", "statefp"], keep="last").to_csv(
            path, index=False
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
