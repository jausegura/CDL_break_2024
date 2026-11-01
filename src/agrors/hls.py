"""Medianas mensuales HLS v2 (L30 + S30) en una ventana de una tesela MGRS, desde Planetary Computer.

Bandas armonizadas: blue, green, red, nir (L30 B05 / S30 B8A), swir1, swir2. Máscara Fmask:
se descartan nube (bit 1), adyacente a nube/sombra (bit 2), sombra (bit 3) y nieve (bit 4).
"""

from __future__ import annotations

import time

import numpy as np
import rasterio
from rasterio.windows import Window

BANDS = {
    "hls2-l30": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B05", "swir1": "B06", "swir2": "B07"},
    "hls2-s30": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B8A", "swir1": "B11", "swir2": "B12"},
}
BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2"]
FMASK_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def search(mgrs: str, lonlat: tuple[float, float], year: int, months: list[int], max_cloud: int):
    """Escenas L30+S30 de la tesela `mgrs` que contienen el punto `lonlat` (lon, lat)."""
    import calendar

    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    last = calendar.monthrange(year, months[-1])[1]
    items = []
    span = f"{year}-{months[0]:02d}-01/{year}-{months[-1]:02d}-{last}"
    for coll in BANDS:
        s = cat.search(
            collections=[coll],
            datetime=span,
            intersects={"type": "Point", "coordinates": list(lonlat)},
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items += [i for i in s.items() if i.id.split(".")[2] == f"T{mgrs}"]
    return sorted(items, key=lambda i: i.datetime)


def read_item(item, win: Window, retries: int = 4) -> np.ndarray | None:
    """(6, h, w) float32 reflectancia con NaN donde Fmask es malo; None si todo está enmascarado."""
    import planetary_computer as pc

    for attempt in range(retries):
        try:
            item = pc.sign(item)
            with rasterio.Env(**GDAL_ENV):
                with rasterio.open(item.assets["Fmask"].href) as f:
                    fm = f.read(1, window=win)
                good = (fm & FMASK_BAD) == 0
                if good.mean() < 0.05:
                    return None
                out = np.full((6, int(win.height), int(win.width)), np.nan, dtype=np.float32)
                for k, name in enumerate(BAND_ORDER):
                    with rasterio.open(item.assets[BANDS[item.collection_id][name]].href) as s:
                        a = s.read(1, window=win).astype(np.float32)
                        a[(a == s.nodata) | (a < -100)] = np.nan
                        out[k] = np.where(good, a * 1e-4, np.nan)
                return out
        except rasterio.errors.RasterioIOError:
            time.sleep(5 * 2**attempt)
    return None


def monthly_medians(items, win: Window, months: list[int], threads: int = 12) -> np.ndarray:
    """(n_meses, 7, h, w): 6 bandas + NDVI, mediana por mes ignorando NaN.

    Las escenas de cada mes se leen en paralelo con hilos (lectura HTTP, limitada por E/S).
    """
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    h, w = int(win.height), int(win.width)
    out = np.full((len(months), 7, h, w), np.nan, dtype=np.float32)
    for mi, m in enumerate(months):
        month_items = [i for i in items if i.datetime.month == m]
        with ThreadPoolExecutor(threads) as ex:
            stack = [a for a in ex.map(lambda i: read_item(i, win), month_items) if a is not None]
        if not stack:
            continue
        arr = np.stack(stack)  # (n, 6, h, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (arr[:, 3] - arr[:, 2]) / (arr[:, 3] + arr[:, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[mi, :6] = np.nanmedian(arr, axis=0)
            out[mi, 6] = np.nanmedian(ndvi, axis=0)
    return out
