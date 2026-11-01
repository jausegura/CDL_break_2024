"""Acceso por ventana a las capas CDL en la rejilla de referencia (condados, 30 m) o a 10 m.

Variantes para los años de la era nueva (≥2024):
  nass30   producto 30 m publicado por NASS (declarado "nearest neighbor"; en 2024 no lo es, ver 02c)
  maj30    mayoría 3x3 desde el 10 m (scripts/02c_majority.py)
  nn30     vecino más próximo verdadero (subpíxel central) del 10 m publicado (scripts/02i_nn30.py);
           en 2025 coincide al 100 % con el producto NASS 30 m, que se usa directamente
  native10 comparación a 10 m; los años ≤2023 se replican 3x3 (cada píxel 30 m = 9 subpíxeles)
"""

from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from agrors.grid import cdl_path, county_path, maj30_path, nn30_path, read_aligned, window10

NEW_ERA_START = 2024
VARIANTS = ("nass30", "maj30", "native10", "nn30")


class Layers:
    """Abre perezosamente los rásteres y devuelve bloques alineados a la rejilla de condados."""

    def __init__(self):
        self._open: dict = {}
        with rasterio.open(county_path()) as z:
            self.t30 = z.transform
            self.shape30 = (z.height, z.width)
        self.t10 = Affine(self.t30.a / 3, 0, self.t30.c, 0, self.t30.e / 3, self.t30.f)

    def _src(self, path):
        key = str(path)
        if key not in self._open:
            self._open[key] = rasterio.open(path)
        return self._open[key]

    def zone(self, win: Window, grid: int = 30) -> np.ndarray:
        z = self._src(county_path()).read(1, window=win)
        return np.repeat(np.repeat(z, 3, 0), 3, 1) if grid == 10 else z

    def cdl(self, year: int, variant: str, win: Window) -> np.ndarray:
        """Bloque CDL de `year` según `variant`; 30 m salvo `native10` (10 m)."""
        shape = (int(win.height), int(win.width))
        if variant == "native10":
            if year >= NEW_ERA_START:
                w10 = window10(win)
                return read_aligned(
                    self._src(cdl_path(year, 10)), self.t10, w10, (shape[0] * 3, shape[1] * 3)
                )
            a = read_aligned(self._src(cdl_path(year, 30)), self.t30, win, shape)
            return np.repeat(np.repeat(a, 3, 0), 3, 1)
        if year >= NEW_ERA_START and variant == "maj30":
            return read_aligned(self._src(maj30_path(year)), self.t30, win, shape)
        if year == NEW_ERA_START and variant == "nn30":
            return read_aligned(self._src(nn30_path(year)), self.t30, win, shape)
        return read_aligned(self._src(cdl_path(year, 30)), self.t30, win, shape)

    def close(self):
        for s in self._open.values():
            s.close()
        self._open.clear()
