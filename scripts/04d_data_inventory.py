"""Fase 4: inventario de datos para la Tabla 1 (recuentos leídos de los ficheros descargados).

Salida: results/tables/data_inventory.csv (dataset, item, value).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.grid import cdl_path  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    rows = []
    for w in cfg["csb"]["windows"]:
        z = ROOT / "data" / "raw" / "csb" / f"NationalCSB_{w}_rev23.zip"
        with zipfile.ZipFile(z) as zf:
            root = zf.namelist()[0].split("/")[0]
        info = pyogrio.read_info(f"/vsizip/{z.as_posix()}/{root}/CSB{w[2:4]}{w[7:9]}.gdb")
        rows.append({"dataset": f"CSB {w}", "item": "national_polygons", "value": int(info["features"])})
    for y, r in [(2023, 30), (2024, 30), (2024, 10), (2025, 30), (2025, 10)]:
        with rasterio.open(cdl_path(y, r)) as s:
            rows.append(
                {"dataset": f"CDL {y} {r} m", "item": "grid_rows_x_cols", "value": f"{s.height} x {s.width}"}
            )
    c = pd.read_csv(ROOT / "data" / "processed" / "counties.csv")
    rows.append({"dataset": "Census counties 2023", "item": "conus_counties", "value": len(c)})
    q = pd.read_parquet(ROOT / "data" / "processed" / "quickstats_county.parquet")
    q = q.dropna(subset=["planted"])
    for crop, g in q[q.year.between(2022, 2025)].groupby("crop"):
        for y, gy in g.groupby("year"):
            rows.append(
                {"dataset": "Quick Stats county planted", "item": f"{crop}_{y}_counties", "value": len(gy)}
            )
    cs = gpd.read_parquet(ROOT / "data" / "interim" / "cropsight" / "cropsight.parquet")
    rows.append({"dataset": "CropSight-US 2021-2023", "item": "objects", "value": len(cs)})
    m = pd.read_csv(ROOT / "data" / "MANIFEST.csv")
    rows.append({"dataset": "all", "item": "manifest_files", "value": len(m)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "data_inventory.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
