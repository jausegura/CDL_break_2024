"""Fase 2i: vecino más próximo verdadero 10 m → 30 m (subpíxel central) para 2024 (variante nn30).

El producto NASS 30 m de 2024 no coincide con el subpíxel central del 10 m publicado (90,7 % CONUS);
en 2025 sí (100 %). Esta variante permite medir la ruptura sin depender del fichero 30 m de 2024.
Salida: data/interim/cdl/cdl_2024_nn30.tif
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import center3  # noqa: E402
from agrors.grid import bounded_map, cdl_path, county_path, nn30_path, read_aligned, window10, windows  # noqa: E402
from agrors.io import load_config  # noqa: E402
from agrors.layers import Layers  # noqa: E402

YEAR = 2024
_L: Layers | None = None


def _init():
    global _L
    _L = Layers()


def _block(win):
    shape = (int(win.height) * 3, int(win.width) * 3)
    a10 = read_aligned(_L._src(cdl_path(YEAR, 10)), _L.t10, window10(win), shape)
    return win, (center3(a10) if a10.any() else None)


def main() -> int:
    cfg = load_config()
    out = nn30_path(YEAR)
    if out.exists():
        return 0
    n = cfg["compute"]["n_workers"]
    with rasterio.open(county_path()) as z:
        prof = z.profile.copy()
        wins = list(windows(z.width, z.height, cfg["compute"]["block_size"]))
    prof.update(dtype="uint8", nodata=0)
    tmp = out.with_suffix(".tif.part")
    with rasterio.open(tmp, "w", **prof) as dst, ProcessPoolExecutor(n, initializer=_init) as ex:
        for win, arr in bounded_map(ex, _block, wins, 2 * n):
            if arr is not None:
                dst.write(arr, 1, window=win)
    tmp.replace(out)
    print("escrito", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
