"""Rejilla nacional CDL (EPSG:5070) y utilidades de lectura por bloques 30 m / 10 m."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from agrors.io import ROOT

INTERIM = ROOT / "data" / "interim"


def cdl_path(year: int, res: int) -> Path:
    """Ruta del GeoTIFF CDL extraído. `res` = 30 (nativo ≤2023 o remuestreo NASS) o 10."""
    return INTERIM / "cdl" / f"cdl_{year}_{res}m.tif"


def cdl10_is_national() -> bool:
    """False si el 10 m local procede del extracto del piloto (sidecar .coverage.json)."""
    return not any(cdl_path(y, 10).with_suffix(".coverage.json").exists() for y in (2024, 2025))


def maj30_path(year: int) -> Path:
    """CDL 10 m agregado a 30 m por mayoría 3x3 (variante propia)."""
    return INTERIM / "cdl" / f"cdl_{year}_maj30.tif"


def nn30_path(year: int) -> Path:
    """Vecino más próximo verdadero (subpíxel central) del 10 m publicado (variante nn30)."""
    return INTERIM / "cdl" / f"cdl_{year}_nn30.tif"


def county_path() -> Path:
    return INTERIM / "zones" / "county30.tif"


def windows(width: int, height: int, size: int) -> Iterator[Window]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def window10(w30: Window) -> Window:
    """Ventana 10 m equivalente a una ventana 30 m (rejillas anidadas, factor 3)."""
    return Window(w30.col_off * 3, w30.row_off * 3, w30.width * 3, w30.height * 3)


def check_nested(path30: Path, path10: Path) -> dict:
    """Comprueba que la rejilla 10 m anida en la de 30 m (mismo CRS, origen común, factor 3)."""
    with rasterio.open(path30) as a, rasterio.open(path10) as b:
        t30, t10 = a.transform, b.transform
        out = {
            "crs_equal": a.crs == b.crs,
            "res30": (t30.a, -t30.e),
            "res10": (t10.a, -t10.e),
            "origin30": (t30.c, t30.f),
            "origin10": (t10.c, t10.f),
            "shape30": (a.height, a.width),
            "shape10": (b.height, b.width),
        }
        out["nested"] = bool(
            out["crs_equal"]
            and np.isclose(t30.a, 3 * t10.a)
            and np.isclose(t30.c, t10.c)
            and np.isclose(t30.f, t10.f)
            and b.width == 3 * a.width
            and b.height == 3 * a.height
        )
    return out


def read_aligned(src: rasterio.io.DatasetReader, ref_transform, win: Window, shape: tuple[int, int]):
    """Lee de `src` la región que en la rejilla de referencia corresponde a `win`.

    Admite rásteres con distinto origen (extensiones nacionales que difieren entre años) siempre
    que compartan resolución y alineación de píxel. Las zonas fuera del ráster se rellenan con 0.
    """
    t = src.transform
    dc = (ref_transform.c - t.c) / t.a
    dr = (ref_transform.f - t.f) / t.e
    if not (np.isclose(dc, round(dc)) and np.isclose(dr, round(dr))):
        raise ValueError("Rejillas no alineadas a nivel de píxel")
    w = Window(win.col_off + round(dc), win.row_off + round(dr), win.width, win.height)
    inside = (
        w.col_off >= 0
        and w.row_off >= 0
        and w.col_off + w.width <= src.width
        and w.row_off + w.height <= src.height
    )
    if inside:
        return src.read(1, window=w, out_shape=shape)
    return src.read(1, window=w, boundless=True, fill_value=0, out_shape=shape)


def bounded_map(executor, fn, items, max_pending: int):
    """Como executor.map pero con a lo sumo `max_pending` tareas en vuelo y resultados en orden de
    finalización (evita acumular en memoria resultados de bloques grandes)."""
    from concurrent.futures import FIRST_COMPLETED, wait

    it = iter(items)
    pending = set()
    for item in it:
        pending.add(executor.submit(fn, item))
        if len(pending) >= max_pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                yield f.result()
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            yield f.result()
