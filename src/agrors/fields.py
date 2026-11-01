"""Estratos de parcela a partir de polígonos CSB rasterizados: distancia al borde y tamaño."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

EDGE_BINS_M = (30.0, 60.0)  # → 0: <30 m, 1: 30–60 m, 2: ≥60 m


def boundary_mask(ids: np.ndarray) -> np.ndarray:
    """Píxeles de polígono (id>0) con algún vecino 4-conexo de id distinto (incluido el 0).

    Los bordes del array cuentan como borde (el llamante debe añadir margen).
    """
    b = np.zeros(ids.shape, dtype=bool)
    b[1:, :] |= ids[1:, :] != ids[:-1, :]
    b[:-1, :] |= ids[:-1, :] != ids[1:, :]
    b[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    b[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    b[0, :] = b[-1, :] = True
    b[:, 0] = b[:, -1] = True
    return b & (ids > 0)


def edge_distance_m(ids: np.ndarray, pixel_m: float) -> np.ndarray:
    """Distancia aproximada (m) del centro de cada píxel de polígono al borde de su polígono.

    = distancia euclídea al centro del píxel de borde más próximo + medio píxel. Fuera de
    polígonos devuelve NaN.
    """
    b = boundary_mask(ids)
    d = distance_transform_edt(~b).astype(np.float32) * pixel_m + pixel_m / 2
    d[ids == 0] = np.nan
    return d


def edge_bin(dist_m: np.ndarray, bins=EDGE_BINS_M) -> np.ndarray:
    """0, 1, 2 según los cortes; 255 fuera de polígono."""
    out = np.digitize(np.nan_to_num(dist_m, nan=-1.0), bins).astype(np.uint8)
    out[np.isnan(dist_m)] = 255
    return out


def size_deciles(acres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decil (0..9) de cada polígono según su superficie y los cortes usados."""
    cuts = np.quantile(acres, np.linspace(0.1, 0.9, 9))
    return np.digitize(acres, cuts).astype(np.uint8), cuts
