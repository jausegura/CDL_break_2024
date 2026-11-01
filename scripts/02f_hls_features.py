"""Fase 2f (E4): medianas mensuales HLS y etiquetas CDL en ventanas de teselas MGRS.

Para cada ventana de config `model_e4.windows` y cada año del piloto:
  data/processed/hls/{ventana}_{año}.npz  →  X (meses, 7, h, w) float16, transform, crs
  data/processed/hls/{ventana}_labels.npz →  cdl_{año}_{nass30|maj30} uint8 en la rejilla HLS
Las etiquetas se reproyectan por vecino más próximo desde la rejilla CDL (EPSG:5070).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.grid import cdl_path, maj30_path  # noqa: E402
from agrors.hls import monthly_medians, search  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402

OUT = ROOT / "data" / "processed" / "hls"


def tile_grid(items):
    """Transform y CRS de la tesela a partir del primer Fmask."""
    import planetary_computer as pc

    with rasterio.open(pc.sign(items[0]).assets["Fmask"].href) as f:
        return f.transform, f.crs


def labels(win_t, crs, shape, years) -> dict[str, np.ndarray]:
    out = {}
    for y in years:
        sources = {"nass30": cdl_path(y, 30)}
        if y >= 2024:
            sources["maj30"] = maj30_path(y)
        for v, p in sources.items():
            if not p.exists():
                continue
            dst = np.zeros(shape, dtype=np.uint8)
            with rasterio.open(p) as src:
                reproject(
                    rasterio.band(src, 1),
                    dst,
                    dst_transform=win_t,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=0,
                )
            out[f"cdl_{y}_{v}"] = dst
    return out


def main() -> int:
    cfg = load_config()
    e4 = cfg["model_e4"]
    OUT.mkdir(parents=True, exist_ok=True)
    years = cfg["pilot"]["years"]
    for w in e4["windows"]:
        win = Window(w["col_off"], w["row_off"], w["size"], w["size"])
        grid = None
        for y in years:
            dest = OUT / f"{w['name']}_{y}.npz"
            if dest.exists():
                continue
            t0 = time.time()
            items = search(w["mgrs"], tuple(w["lonlat"]), y, e4["months"], e4["max_cloud"])
            grid = grid or tile_grid(items)
            X = monthly_medians(items, win, e4["months"])
            wt = win_transform(win, grid[0])
            np.savez_compressed(
                dest, X=X.astype(np.float16), transform=np.array(wt)[:6], crs=str(grid[1]), n_items=len(items)
            )
            valid = np.isfinite(X[:, 6]).mean(axis=(1, 2))
            print(
                w["name"],
                y,
                len(items),
                "escenas; válidos por mes",
                np.round(valid, 2),
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
        lab = OUT / f"{w['name']}_labels.npz"
        if grid is None:
            d = np.load(OUT / f"{w['name']}_{years[0]}.npz")
            t = rasterio.Affine(*d["transform"])
            crs = rasterio.crs.CRS.from_string(str(d["crs"]))
        else:
            t, crs = win_transform(win, grid[0]), grid[1]
        lbl = labels(t, crs, (w["size"], w["size"]), years)
        np.savez_compressed(lab, **lbl)
        print(w["name"], "etiquetas:", sorted(lbl), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
