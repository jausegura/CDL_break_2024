import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import (  # noqa: E402
    GROUP_ORDER,
    GROUPS,
    UNGROUPED,
    center3,
    group_lut,
    majority3,
    merge_counts,
    transition_counts,
)


def test_groups_disjoint_and_cover_focus():
    seen = set()
    for codes in GROUPS.values():
        assert not (seen & codes)
        seen |= codes
    for code in (1, 2, 5, 24, 36, 61, 176, 141, 121, 111):
        assert code in seen


def test_group_lut():
    lut = group_lut()
    assert lut[1] == 0 and lut[176] == 1 and lut[0] == 255


def test_groups_follow_usda_legend():
    lut = group_lut()
    crop, grass = GROUP_ORDER.index("crop"), GROUP_ORDER.index("grass_pasture")
    for code in (58, 59, 60, 61, 36, 37):  # Clover, Sod/Grass Seed, Switchgrass, fallow, hay: cultivos
        assert lut[code] == crop
    assert lut[62] == grass  # Pasture/Grass (código retirado) = 176
    assert lut[81] == UNGROUPED  # Clouds/No Data no es una cobertura


def test_majority3_basic_and_ties():
    a = np.array([[1, 1, 5], [1, 5, 5], [2, 2, 5]], dtype=np.uint8)
    assert majority3(a)[0, 0] == 5  # 5 aparece 4 veces, 1 tres veces
    tie = np.array([[1, 1, 1], [2, 5, 2], [2, 0, 0]], dtype=np.uint8)
    # 1 y 2 empatan a 3 votos; el central (5) no está empatado → primero en orden de lectura (1)
    assert majority3(tie)[0, 0] == 1
    tie_c = np.array([[1, 1, 3], [2, 2, 3], [1, 2, 0]], dtype=np.uint8)
    assert majority3(tie_c)[0, 0] == 2  # 1 y 2 empatan a 3; el central es 2
    assert majority3(np.zeros((3, 3), dtype=np.uint8))[0, 0] == 0


def test_majority3_shape():
    a = np.random.default_rng(42).integers(0, 4, size=(9, 12), dtype=np.uint8)
    assert majority3(a).shape == (3, 4)
    assert center3(a).shape == (3, 4)
    assert center3(a)[0, 0] == a[1, 1]


def test_transition_counts_and_merge():
    zone = np.array([[1, 1], [2, 0]], dtype=np.uint16)
    a = np.array([[1, 1], [5, 5]], dtype=np.uint8)
    b = np.array([[5, 1], [5, 5]], dtype=np.uint8)
    t = transition_counts(zone, a, b)
    d = {tuple(r[:3]): r[3] for r in t}
    assert d == {(1, 1, 5): 1, (1, 1, 1): 1, (2, 5, 5): 1}
    m = merge_counts([t, t])
    assert m[:, 3].sum() == 6


def test_transition_counts_nodata():
    zone = np.ones((1, 3), dtype=np.uint16)
    a = np.array([[0, 1, 1]], dtype=np.uint8)
    b = np.array([[1, 0, 1]], dtype=np.uint8)
    t = transition_counts(zone, a, b)
    assert t.tolist() == [[1, 1, 1, 1]]
