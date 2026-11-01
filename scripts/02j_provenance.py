"""Fase 2j: procedencia y alineación de los ficheros CDL de la era nueva (2024–2025).

1. Perfil de cada ráster nacional usado (CRS, transformación, origen, tamaño, nodata, tipo) y sha256 del zip
   oficial del que procede (data/MANIFEST.csv) → results/tables/provenance_rasters.csv
2. En bloques aleatorios de CONUS (semilla de la configuración; 512 × 512 píxeles de 30 m con ≥ 50 % de
   píxeles con dato en el producto NASS 30 m), acuerdo del producto NASS 30 m con cada uno de los nueve
   subpíxeles del 10 m oficial (desplazamientos fila/columna 0–2) y con la mayoría 3 × 3; acuerdo en los
   bloques 3 × 3 homogéneos del 10 m (donde cualquier remuestreo del mismo mapa debe coincidir) y
   coherencia de nodata → results/tables/provenance_offsets.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import majority3  # noqa: E402
from agrors.grid import cdl_path, read_aligned, window10  # noqa: E402
from agrors.io import ROOT, load_config, read_manifest  # noqa: E402
from agrors.layers import Layers  # noqa: E402

N_BLOCKS = 100
SIZE = 512
OUT = ROOT / "results" / "tables"


def profiles(years: list[int]) -> pd.DataFrame:
    man = read_manifest()
    rows = []
    for y in years:
        for res in (30, 10):
            p = cdl_path(y, res)
            with rasterio.open(p) as s:
                t = s.transform
                zip_rel = f"data/raw/cdl/{y}_{res}m_cdls.zip"
                rows.append(
                    {
                        "year": y,
                        "res_m": res,
                        "file": p.relative_to(ROOT).as_posix(),
                        "crs": s.crs.to_string(),
                        "pixel_x": t.a,
                        "pixel_y": t.e,
                        "origin_x": t.c,
                        "origin_y": t.f,
                        "width": s.width,
                        "height": s.height,
                        "nodata": s.nodata,
                        "dtype": s.dtypes[0],
                        "source_zip": zip_rel,
                        "zip_sha256": man.get(zip_rel, {}).get("sha256", ""),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    years = cfg["sources"]["cdl_10m_years"]
    profiles(years).to_csv(OUT / "provenance_rasters.csv", index=False)
    rng = np.random.default_rng(cfg["seed"])
    L = Layers()
    rows = []
    for y in years:
        with rasterio.open(cdl_path(y, 10)) as s10, rasterio.open(cdl_path(y, 30)) as s30:
            got = tried = 0
            while got < N_BLOCKS and tried < 20000:
                tried += 1
                r = int(rng.integers(0, L.shape30[0] - SIZE))
                c = int(rng.integers(0, L.shape30[1] - SIZE))
                w = Window(c, r, SIZE, SIZE)
                if (L.zone(w) > 0).mean() < 0.5:
                    continue
                n30 = read_aligned(s30, L.t30, w, (SIZE, SIZE))
                m = n30 > 0
                if m.mean() < 0.5:
                    continue
                a = read_aligned(s10, L.t10, window10(w), (3 * SIZE, 3 * SIZE))
                b = a.reshape(SIZE, 3, SIZE, 3)
                homog = (b == b[:, :1, :, :1]).all(axis=(1, 3)) & (b[:, 0, :, 0] > 0)
                got += 1
                row = {"year": y, "col30": c, "row30": r, "n_valid": int(m.sum())}
                for dr in range(3):
                    for dc in range(3):
                        row[f"agree_r{dr}c{dc}"] = float((n30[m] == a[dr::3, dc::3][m]).mean())
                row["agree_majority"] = float((n30[m] == majority3(a)[m]).mean())
                hm = m & homog
                row["share_homogeneous"] = float(hm.sum() / m.sum())
                row["agree_homogeneous"] = (
                    float((n30[hm] == b[:, 0, :, 0][hm]).mean()) if hm.any() else np.nan
                )
                row["nodata_mismatch"] = float(((n30 == 0) != (a[1::3, 1::3] == 0)).mean())
                rows.append(row)
    L.close()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "provenance_offsets.csv", index=False)
    print(d.groupby("year").mean(numeric_only=True).drop(columns=["col30", "row30"]).round(4).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
