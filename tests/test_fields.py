import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.fields import boundary_mask, edge_bin, edge_distance_m, size_deciles  # noqa: E402


def test_boundary_and_distance_square_field():
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 7  # campo de 16x16 píxeles de 10 m
    b = boundary_mask(ids)
    assert b[2, 5] and b[17, 10] and not b[5, 5] and not b[0, 0]
    d = edge_distance_m(ids, 10.0)
    assert np.isnan(d[0, 0])
    assert d[2, 8] == 5.0  # píxel de borde: medio píxel
    assert d[5, 9] == 35.0  # 3 píxeles hacia dentro
    bins = edge_bin(d)
    assert bins[2, 8] == 0 and bins[5, 9] == 1 and bins[9, 9] == 2 and bins[0, 0] == 255


def test_adjacent_fields_share_boundary():
    ids = np.zeros((10, 22), dtype=np.int32)
    ids[1:9, 1:11] = 1
    ids[1:9, 11:21] = 2
    b = boundary_mask(ids)
    assert b[4, 10] and b[4, 11] and not b[4, 5]


def test_size_deciles():
    acres = np.arange(1, 101, dtype=float)
    dec, cuts = size_deciles(acres)
    assert dec.min() == 0 and dec.max() == 9 and len(cuts) == 9
    assert (np.bincount(dec) == 10).all()
