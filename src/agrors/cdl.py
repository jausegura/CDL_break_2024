"""Códigos CDL, agrupaciones y operaciones por bloque (mayoría 3x3, tablas de transición)."""

from __future__ import annotations

import numpy as np

# Códigos CDL de cultivos de interés (USDA NASS CDL legend).
CORN, COTTON, SOYBEANS, WINTER_WHEAT = 1, 2, 5, 24
DBL_WW_SOY = 26
FOCUS = {"corn": CORN, "soybeans": SOYBEANS, "winter_wheat": WINTER_WHEAT, "cotton": COTTON}
# Para comparar superficie con Quick Stats (superficie sembrada) se incluyen los dobles cultivos
# que contienen el cultivo: 26 WinWht/Soy, 225 WinWht/Corn, 226 Oats/Corn, 228 Triticale/Corn,
# 232 Lettuce/Cotton, 236 WinWht/Sorghum, 237 Barley/Corn, 238 WinWht/Cotton, 239 Soy/Cotton,
# 240 Soy/Oats, 241 Corn/Soy, 254 Barley/Soy.
AREA_CODES = {
    "corn": (1, 225, 226, 228, 237, 241),
    "soybeans": (5, 26, 239, 240, 241, 254),
    "winter_wheat": (24, 26, 225, 236, 238),
    "cotton": (2, 232, 238, 239),
    "hay": (36, 37),  # alfalfa + otro heno; se compara con la superficie cosechada de Quick Stats
}
# Sensibilidad: solo la clase de cultivo único (sin dobles cultivos) frente a la superficie sembrada.
AREA_CODES_SINGLE = {"corn": (1,), "soybeans": (5,), "winter_wheat": (24,), "cotton": (2,)}

# Grupos de cobertura según la leyenda USDA NASS: "crop" = códigos 1–61 (incluye 58 Clover/Wildflowers,
# 59 Sod/Grass Seed, 60 Switchgrass, heno/alfalfa 36–37 y barbecho 61), 66–77 y 204–254. 62 Pasture/Grass
# (código retirado en la recodificación de 2014, sustituido por 176) va con el pastizal. 81 Clouds/No Data
# no es una cobertura: no pertenece a ningún grupo y se trata como sin dato (ver UNGROUPED).
GROUPS = {
    "crop": set(range(1, 62)) | set(range(66, 78)) | set(range(204, 255)),
    "grass_pasture": {62, 176},
    "forest_shrub": {63, 64, 141, 142, 143, 152},
    "developed": {82, 121, 122, 123, 124},
    "water": {83, 92, 111, 112},
    "wetland": {87, 190, 195},
    "barren_other": {65, 88, 131},
}
UNGROUPED = 255  # índice de grupo de los códigos sin cobertura (0 fondo, 81 nubes/sin dato, sin uso)
GROUP_ORDER = list(GROUPS)


def group_lut(groups: dict[str, set[int]] | None = None) -> np.ndarray:
    """Tabla de 256 entradas código CDL → índice de grupo (orden de GROUP_ORDER); UNGROUPED = sin dato/fondo.

    `groups` permite construir la tabla de otra asignación de clases con los mismos nombres de grupo.
    """
    groups = GROUPS if groups is None else groups
    lut = np.full(256, UNGROUPED, dtype=np.uint8)
    for i, name in enumerate(GROUP_ORDER):
        for code in groups[name]:
            lut[code] = i
    return lut


def majority3(a: np.ndarray, nodata: int = 0) -> np.ndarray:
    """Mayoría en bloques 3x3 no solapados (10 m → 30 m).

    Empates: gana el valor del subpíxel central si está entre los empatados; si no, el primero
    en orden de lectura (fila a fila). Los subpíxeles `nodata` no votan; si los nueve lo son,
    el resultado es `nodata`.
    """
    h, w = a.shape
    if h % 3 or w % 3:
        raise ValueError("Las dimensiones deben ser múltiplos de 3")
    # (h/3, w/3, 9) con el central (índice 4) movido a la posición 0 para el desempate.
    b = a.reshape(h // 3, 3, w // 3, 3).transpose(0, 2, 1, 3).reshape(h // 3, w // 3, 9)
    order = [4, 0, 1, 2, 3, 5, 6, 7, 8]
    b = b[:, :, order]
    valid = b != nodata
    counts = np.zeros(b.shape, dtype=np.uint8)
    for k in range(9):
        counts[:, :, k] = ((b == b[:, :, k : k + 1]) & valid).sum(axis=2)
    counts[~valid] = 0
    idx = counts.argmax(axis=2)  # argmax devuelve el primer máximo → central primero
    out = np.take_along_axis(b, idx[:, :, None], axis=2)[:, :, 0]
    out[counts.max(axis=2) == 0] = nodata
    return out


def center3(a: np.ndarray) -> np.ndarray:
    """Subpíxel central de cada bloque 3x3 (vecino más próximo 10 m → 30 m)."""
    return a[1::3, 1::3]


def transition_counts(zone: np.ndarray, a: np.ndarray, b: np.ndarray, nodata: int = 0) -> np.ndarray:
    """Recuentos de transición (zona, clase_a, clase_b) para píxeles con dato en ambos años.

    Devuelve un array (n, 4) int64 con columnas zone, from, to, count. `zone` = 0 se descarta.
    """
    m = (zone > 0) & (a != nodata) & (b != nodata)
    if not m.any():
        return np.empty((0, 4), dtype=np.int64)
    key = (zone[m].astype(np.int64) << 16) | (a[m].astype(np.int64) << 8) | b[m].astype(np.int64)
    uniq, cnt = np.unique(key, return_counts=True)
    return np.column_stack([uniq >> 16, (uniq >> 8) & 0xFF, uniq & 0xFF, cnt])


def merge_counts(parts: list[np.ndarray]) -> np.ndarray:
    """Suma recuentos (zone, from, to, count) procedentes de varios bloques."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.empty((0, 4), dtype=np.int64)
    allp = np.concatenate(parts)
    key = (allp[:, 0] << 16) | (allp[:, 1] << 8) | allp[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=allp[:, 3]).astype(np.int64)
    return np.column_stack([uniq >> 16, (uniq >> 8) & 0xFF, uniq & 0xFF, cnt])
