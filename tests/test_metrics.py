import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.metrics import bootstrap_ratio_excess, components, excess, rates  # noqa: E402


def _t(rows):
    return pd.DataFrame(rows, columns=["zone", "from", "to", "n"])


def test_components_and_rates():
    # zona 1: 6 maíz→soja, 2 maíz→maíz, 1 maíz→pasto (176), 1 bosque(141)→bosque
    t = _t([(1, 1, 5, 6), (1, 1, 1, 2), (1, 1, 176, 1), (1, 141, 141, 1)])
    r = rates(components(t)).iloc[0]
    assert r.n == 10
    assert np.isclose(r.any_change, 7 / 10)
    assert np.isclose(r.group_change, 1 / 10)
    assert np.isclose(r.crop_gross, 1 / 9)
    assert np.isclose(r.crop_net, (8 - 9) / 9)
    assert np.isclose(r.crop_type_change, 6 / 8)
    assert np.isclose(r.nonag_change, 0.0)
    assert np.isclose(r.corn_corn, 2 / 9)
    assert np.isclose(r.cs_alternation, 6 / 9)
    assert r.area0_corn == 9 and r.area1_soybeans == 6


def test_components_additive_across_zones():
    t = _t([(1, 1, 5, 3), (2, 1, 1, 1), (2, 5, 1, 2)])
    c = components(t)
    tot = components(t.assign(zone=0))
    assert np.isclose(c.drop(columns="zone").sum().to_numpy(), tot.drop(columns="zone").to_numpy()).all()


def test_excess():
    assert np.isclose(excess(pd.Series([0.3]), pd.Series([0.1]), pd.Series([0.2])).iloc[0], 0.15)


def test_bootstrap_constant_when_identical_zones():
    idx = pd.Index([1, 2, 3], name="zone")
    mk = lambda r: pd.DataFrame({"num": [r * 10] * 3, "den": [10.0] * 3}, index=idx)  # noqa: E731
    b = bootstrap_ratio_excess(
        {"before": mk(0.1), "break": mk(0.4), "after": mk(0.2)}, "num", "den", reps=50, seed=42
    )
    assert np.allclose(b, 0.4 - 0.15)


def test_bootstrap_stats():
    from agrors.metrics import bootstrap_ratio_stats

    idx = pd.Index([1, 2, 3], name="zone")
    mk = lambda r: pd.DataFrame({"num": [r * 10] * 3, "den": [10.0] * 3}, index=idx)  # noqa: E731
    b = bootstrap_ratio_stats(
        {"before": mk(0.1), "break": mk(0.4), "after": mk(0.3)}, "num", "den", reps=20, seed=42
    )
    assert np.allclose(b["excess"], 0.2) and np.allclose(b["step"], 0.3) and np.allclose(b["persist"], 0.2)


def test_bootstrap_cluster_resamples_whole_groups():
    from agrors.metrics import bootstrap_ratio_stats

    idx = pd.Index([1, 2, 3, 4], name="zone")
    before = pd.DataFrame({"num": [1.0, 1.0, 1.0, 1.0], "den": [10.0] * 4}, index=idx)
    brk = pd.DataFrame({"num": [5.0, 5.0, 1.0, 1.0], "den": [10.0] * 4}, index=idx)
    clusters = pd.Series(["A", "A", "B", "B"], index=idx)
    b = bootstrap_ratio_stats(
        {"before": before, "break": brk, "after": before}, "num", "den", 200, 42, clusters=clusters
    )
    # con grupos completos el salto solo puede valer 0 (solo B), 0,2 (A y B) o 0,4 (solo A)
    assert set(np.round(b["step"], 6)) <= {0.0, 0.2, 0.4}


def test_components_drop_no_data_codes():
    # 81 (nubes/sin dato) no cuenta en ningún denominador ni numerador
    t = _t([(1, 1, 5, 6), (1, 1, 81, 3), (1, 81, 176, 2)])
    r = rates(components(t)).iloc[0]
    assert r.n == 6 and np.isclose(r.crop_gross, 0.0)


def test_components_alternative_lut():
    from agrors.cdl import GROUPS, group_lut

    old = {k: set(v) for k, v in GROUPS.items()}
    old["crop"] -= {59}
    old["grass_pasture"] |= {59}
    t = _t([(1, 1, 59, 4), (1, 1, 1, 6)])
    assert np.isclose(rates(components(t)).iloc[0].crop_gross, 0.0)
    assert np.isclose(rates(components(t, lut=group_lut(old))).iloc[0].crop_gross, 0.4)
