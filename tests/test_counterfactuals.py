import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cf03j", ROOT / "scripts" / "03j_counterfactuals.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def test_decomposition_recovers_model_terms():
    # r = R + N_era + ν·S + I·B; el par de la ruptura lleva la media del ruido de ambas eras
    R, n_old, n_new, nu, inc = 5.0, 2.0, 4.0, 1.0, 3.0
    same = [2014, 2015, 2016, 2017, 2019, 2021, 2022]
    switch = [2018, 2020]
    rate = {y: R + n_old for y in same} | {y: R + n_old + nu for y in switch}
    rate[2023] = R + (n_old + n_new) / 2 + nu + inc
    rate[2024] = R + n_new + nu
    e = cf.estimates(rate, same, switch)
    assert np.isclose(e["decomp_nlcd_nu"], nu)
    assert np.isclose(e["persist_minus_old_nlcd_switch"], n_new - n_old)
    assert np.isclose(e["decomp_inconsistency_I"], inc)
    assert np.isclose(e["break_minus_adjacent"], inc + nu / 2)  # exceso principal = I + ½ν
    assert np.isclose(e["break_minus_trend"], e["break_minus_old_same_nlcd"])  # sin tendencia
