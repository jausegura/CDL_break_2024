"""Matrices de error oficiales de NASS (ficheros NASS_CDL_{ST}{YY}_accuracy.xlsx de
CDL_{año}_accuracy_assessments.zip): hoja "All - Matrix", filas = clase del mapa, columnas = clase de
referencia, códigos CDL 1–256 (la última fila y columnas de totales se descartan)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import openpyxl

SHEET = "All - Matrix"


def read_matrix(source) -> np.ndarray:
    """Matriz 256 × 256 (índice = código CDL; filas mapa, columnas referencia) de un xlsx (ruta o bytes)."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
    rows = list(wb[SHEET].iter_rows(values_only=True))
    wb.close()
    header = rows[0]
    cols = {j: int(v) for j, v in enumerate(header) if isinstance(v, int | float) and 0 <= v < 256}
    m = np.zeros((256, 256), dtype=np.float64)
    for r in rows[1:]:
        code = r[1]
        if not isinstance(code, int | float) or not 0 <= code < 256:
            continue
        for j, c in cols.items():
            v = r[j]
            if isinstance(v, int | float):
                m[int(code), c] = v
    return m


def state_matrices(zip_path: Path, year: int) -> dict[str, np.ndarray]:
    """Matrices de todos los estados de un zip anual: {abreviatura del estado: matriz}."""
    out = {}
    yy = f"{year % 100:02d}"
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            stem = Path(name).stem  # NASS_CDL_IA24_accuracy
            if not stem.startswith("NASS_CDL_") or not stem.endswith(f"{yy}_accuracy"):
                continue
            code = stem.split("_")[2]  # "IA24" o "National24"
            key = "NATIONAL" if code.lower().startswith("national") else code[:2]
            out[key] = read_matrix(z.read(name))
    return out


def group_matrix(m: np.ndarray, lut: np.ndarray, n_groups: int, ungrouped: int) -> np.ndarray:
    """Agrega la matriz por grupos de cobertura (filas mapa, columnas referencia).

    Los códigos sin grupo (fondo, 81) se descartan."""
    g = np.zeros((n_groups, n_groups))
    rows, cols = np.nonzero(m)
    for i, j in zip(rows, cols, strict=True):
        if lut[i] != ungrouped and lut[j] != ungrouped:
            g[lut[i], lut[j]] += m[i, j]
    return g
