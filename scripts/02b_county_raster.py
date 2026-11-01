"""Fase 2b: rasteriza los condados (Census cartographic boundary 2023) en la rejilla CDL 30 m.

Salidas:
  data/interim/zones/county30.tif   uint16, índice de condado (1..N), 0 fuera de CONUS
  data/processed/counties.csv       índice → GEOID, estado, nombre, área (km²)
Referencia de rejilla: CDL 2021 30 m.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import transform as win_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.grid import bounded_map, cdl_path, county_path, windows  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402

NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
REF_YEAR = 2021  # todas las CDL nacionales comparten la rejilla estándar (read_aligned admite desfases)
_G = None


def load_counties() -> gpd.GeoDataFrame:
    zf = ROOT / "data" / "raw" / "boundaries" / "cb_2023_us_county_500k.zip"
    g = gpd.read_file(f"zip://{zf.as_posix()}")
    g = g[~g["STATEFP"].isin(NON_CONUS)].sort_values("GEOID").reset_index(drop=True)
    g["zone"] = np.arange(1, len(g) + 1, dtype=np.uint16)
    with rasterio.open(cdl_path(REF_YEAR, 30)) as src:
        g = g.to_crs(src.crs)
    return g[["zone", "GEOID", "STATEFP", "STUSPS", "NAME", "geometry"]]


def _init(gpkg: str):
    global _G
    _G = gpd.read_file(gpkg)


def _block(args):
    win, transform = args
    t = win_transform(win, transform)
    h, w = int(win.height), int(win.width)
    x0, y0 = t.c, t.f
    bbox = (x0, y0 + t.e * h, x0 + t.a * w, y0)
    sub = _G.iloc[list(_G.sindex.intersection(bbox))]
    if sub.empty:
        return win, None
    arr = rasterize(
        zip(sub.geometry, sub.zone, strict=True), out_shape=(h, w), transform=t, fill=0, dtype="uint16"
    )
    return win, arr


def main() -> int:
    cfg = load_config()
    n_workers = cfg["compute"]["n_workers"]
    out = county_path()
    if out.exists():
        print("ya existe", out)
        return 0
    g = load_counties()
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    tab = g.drop(columns="geometry").copy()
    tab["area_km2"] = g.geometry.area / 1e6
    tab.to_csv(proc / "counties.csv", index=False)
    tmp_gpkg = ROOT / "data" / "interim" / "zones" / "counties_5070.gpkg"
    tmp_gpkg.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(tmp_gpkg, driver="GPKG")
    with rasterio.open(cdl_path(REF_YEAR, 30)) as src:
        prof = src.profile.copy()
        transform = src.transform
        wins = list(windows(src.width, src.height, cfg["compute"]["block_size"]))
    prof.update(
        dtype="uint16",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
        count=1,
    )
    prof.pop("photometric", None)
    tmp = out.with_suffix(".tif.part")
    with (
        rasterio.open(tmp, "w", **prof) as dst,
        ProcessPoolExecutor(cfg["compute"]["n_workers"], initializer=_init, initargs=(str(tmp_gpkg),)) as ex,
    ):
        for win, arr in bounded_map(ex, _block, [(w, transform) for w in wins], 2 * n_workers):
            if arr is not None:
                dst.write(arr, 1, window=win)
    tmp.replace(out)
    print("escrito", out, len(g), "condados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
