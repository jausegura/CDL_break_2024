import io
import sys
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.accuracy import group_matrix, read_matrix  # noqa: E402
from agrors.cdl import GROUP_ORDER, UNGROUPED, group_lut  # noqa: E402


def _workbook() -> bytes:
    """Hoja "All - Matrix" como la de NASS: fila 1 = códigos de referencia y columnas de totales;
    columna B = código del mapa."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "All - Matrix"
    ws.append([None, None, 1, 5, 176, "Total", "User"])
    ws.append([None, 1, 90, 5, 2, 97, 0.93])  # mapa maíz
    ws.append([None, 5, 8, 80, 0, 88, 0.91])  # mapa soja
    ws.append([None, 176, 2, 0, 50, 52, 0.96])  # mapa pastizal
    ws.append([None, "Total", 100, 85, 52, None, None])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_read_matrix_orientation_and_totals():
    m = read_matrix(_workbook())
    assert m.shape == (256, 256)
    assert m[1, 1] == 90 and m[5, 1] == 8 and m[176, 1] == 2  # columna = referencia maíz
    assert m[:, 1].sum() == 100 and m[1].sum() == 97  # totales de referencia y de mapa
    assert m.sum() == 237  # la fila "Total" y las columnas de totales no entran


def test_group_matrix_aggregates_crop_classes():
    g = group_matrix(read_matrix(_workbook()), group_lut(), len(GROUP_ORDER), UNGROUPED)
    crop, grass = GROUP_ORDER.index("crop"), GROUP_ORDER.index("grass_pasture")
    assert g[crop, crop] == 90 + 5 + 8 + 80
    assert g[grass, crop] == 2 and g[crop, grass] == 2 and g[grass, grass] == 50
    assert np.isclose(g.sum(), 237)
