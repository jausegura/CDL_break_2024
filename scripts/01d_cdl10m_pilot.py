"""Fase 1d: CDL 10 m 2024–2025 solo para los estados del piloto, desde la copia Icechunk/Zarr de
Source Cooperative (chill/usda-cropland-data-layer, v0.1.0), que declara píxeles idénticos a los
zips oficiales de NASS. Se usa porque la descarga nacional de 10 m (19 GB) es muy lenta en esta red.

Escribe data/interim/cdl/cdl_{año}_10m.tif (GeoTIFF disperso en la rejilla del almacén, solo con
los bloques que intersectan condados del piloto) y un sidecar .coverage.json = {"coverage": "pilot"}.
La fidelidad se comprueba en 02c (NASS 30 m frente al subpíxel central) y aquí comparando el grupo
30 m del almacén con el zip oficial 30 m en una muestra de bloques.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.grid import cdl_path, county_path  # noqa: E402
from agrors.io import _LOCK, ROOT, load_config, read_manifest, write_manifest  # noqa: E402

BLOCK = 3072  # 6x6 trozos internos de 512
STORE = dict(
    bucket="chill",
    prefix="usda-cropland-data-layer/v0.1.0.icechunk",
    endpoint_url="https://data.source.coop",
    region="us-east-1",
    anonymous=True,
    force_path_style=True,
)


def open_group(group: str):
    import icechunk
    import xarray as xr

    repo = icechunk.Repository.open(icechunk.s3_storage(**STORE))
    return xr.open_zarr(repo.readonly_session("main").store, group=group, chunks=None)


def transform_of(ds) -> Affine:
    gt = [float(v) for v in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


def pilot_blocks(t10: Affine, shape10: tuple[int, int], allowed: np.ndarray) -> list[Window]:
    """Bloques 10 m que intersectan algún condado del piloto (se mira el ráster de condados 30 m)."""
    out = []
    with rasterio.open(county_path()) as z:
        t30 = z.transform
        for r in range(0, shape10[0], BLOCK):
            for c in range(0, shape10[1], BLOCK):
                x0, y0 = t10 * (c, r)
                col30 = round((x0 - t30.c) / t30.a)
                row30 = round((y0 - t30.f) / t30.e)
                w30 = Window(col30, row30, BLOCK // 3 + 1, BLOCK // 3 + 1)
                if col30 + w30.width < 0 or row30 + w30.height < 0 or col30 > z.width or row30 > z.height:
                    continue
                zz = z.read(1, window=w30, boundless=True, fill_value=0)
                if np.isin(zz, allowed).any():
                    out.append(Window(c, r, min(BLOCK, shape10[1] - c), min(BLOCK, shape10[0] - r)))
    return out


def verify_30m(cfg, n_blocks: int = 40) -> dict:
    """Compara el grupo 30 m del almacén con el zip oficial (2024 y 2025) en bloques aleatorios."""
    ds = open_group("30m")
    t = transform_of(ds)
    rng = np.random.default_rng(cfg["seed"])
    res = {}
    for year in (2024, 2025):
        arr = ds["crop_type"].sel(year=year)
        with rasterio.open(cdl_path(year, 30)) as src:
            eq = tot = 0
            for _ in range(n_blocks):
                r = int(rng.integers(0, src.height - 1024))
                c = int(rng.integers(0, src.width - 1024))
                a = src.read(1, window=Window(c, r, 1024, 1024))
                x0, y0 = src.transform * (c, r)
                cs, rs = round((x0 - t.c) / t.a), round((y0 - t.f) / t.e)
                b = arr.isel(y=slice(rs, rs + 1024), x=slice(cs, cs + 1024)).values
                eq += int((a == b).sum())
                tot += a.size
        res[year] = eq / tot
    return res


def main() -> int:
    cfg = load_config()
    counties = pd.read_csv(ROOT / "data" / "processed" / "counties.csv")
    allowed = counties.loc[counties.STUSPS.isin(cfg["pilot"]["states"]), "zone"].to_numpy()
    ver = verify_30m(cfg)
    print("acuerdo almacén 30 m vs zip oficial:", ver, flush=True)
    if min(ver.values()) < 0.999999:
        raise RuntimeError("El almacén Icechunk no coincide con el zip oficial a 30 m")
    ds = open_group("10m")
    t10 = transform_of(ds)
    shape10 = (ds.sizes["y"], ds.sizes["x"])
    blocks = pilot_blocks(t10, shape10, allowed)
    print(len(blocks), "bloques 10 m del piloto", flush=True)
    crs = rasterio.crs.CRS.from_epsg(5070)
    for year in cfg["sources"]["cdl_10m_years"]:
        out = cdl_path(year, 10)
        if out.exists():
            continue
        arr = ds["crop_type"].sel(year=year)
        prof = dict(
            driver="GTiff",
            width=shape10[1],
            height=shape10[0],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=t10,
            nodata=0,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            BIGTIFF="YES",
            SPARSE_OK="TRUE",
        )
        tmp = out.with_suffix(".tif.part")

        def read(w, arr=arr):
            return w, arr.isel(
                y=slice(w.row_off, w.row_off + w.height), x=slice(w.col_off, w.col_off + w.width)
            ).values

        with rasterio.open(tmp, "w", **prof) as dst, ThreadPoolExecutor(6) as ex:
            for i, (w, a) in enumerate(ex.map(read, blocks)):
                if a.any():
                    dst.write(a, 1, window=w)
                if i % 100 == 0:
                    print(year, i, "/", len(blocks), flush=True)
        tmp.replace(out)
        out.with_suffix(".coverage.json").write_text(
            json.dumps(
                {
                    "coverage": "pilot",
                    "states": cfg["pilot"]["states"],
                    "source": "icechunk s3://chill/usda-cropland-data-layer/v0.1.0.icechunk group 10m",
                }
            )
        )
        rel = out.relative_to(ROOT).as_posix()
        with _LOCK:
            m = read_manifest()
            m[rel] = {
                "file": rel,
                "source_url": "https://data.source.coop/chill/usda-cropland-data-layer/"
                "v0.1.0.icechunk (group 10m)",
                "download_date": dt.date.today().isoformat(),
                "sha256": "",
                "license": "Public domain (USDA NASS)",
                "notes": f"CDL {year} 10 m, solo estados piloto; almacén 30 m verificado = zip oficial "
                f"({ver[year]:.6f})",
            }
            write_manifest(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
