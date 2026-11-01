"""Fase 3j (E1, robustez del estimador): contrafactuales alternativos y ajuste por edición NLCD.

Modelo aditivo para la tasa nacional r_p de cada par p = (t, t+1):
    r_p = R_p + N_e(p) + ν·S_p + I·B_p + ε_p
R_p cambio real; N_e error de clasificación propio de cada era (e = antigua a / nueva n); S_p = 1 si cambia la
edición NLCD de entrenamiento; B_p = 1 en el par que cruza la ruptura (2023→24); ε_p ruido del par.

Para la ruptura se compara r(2023→24) con varios contrafactuales c, y se informa r(2023→24) − c:
  adjacent        ½[r(2022→23) + r(2024→25)]       (estimador principal, "exceso")
  pre_break       r(2022→23)                        (salto)
  old_same_nlcd   media de los pares 2014–2023 sin cambio NLCD
  old_nlcd_switch media de los pares 2014–2023 con cambio NLCD (2018→19, 2020→21): contrafactual "solo
                  cambio NLCD"; la diferencia descuenta ν si el efecto NLCD de 2024 es como el de esos pares
  trend           tendencia lineal de 2014–2023 (pares sin cambio NLCD) extrapolada a 2023→24
Para la persistencia (r(2024→25), que también tiene cambio NLCD) se informa r(2024→25) − c con
c = r(2022→23) y c = old_nlcd_switch.

Descomposición. Si el par que cruza la ruptura lleva la media del ruido de ambas eras, ½(N_a + N_n), y el
cambio real y ν son iguales en los pares de referencia y en la ruptura:
  ν (efecto de un cambio de edición NLCD)  = media(old_nlcd_switch) − media(old_same_nlcd)
  ΔN = N_n − N_a (inestabilidad propia del producto nuevo) = r(2024→25) − media(old_nlcd_switch)
  I (inconsistencia entre eras)            = [r(2023→24) − media(old_nlcd_switch)] − ½ ΔN
El exceso principal equivale a I + ½ν bajo los mismos supuestos.

IC al 95 % por percentiles de un bootstrap de condados (remuestreo con reemplazo, probabilidad igual,
cocientes de sumas recalculados en cada réplica). Variante NASS 30 m, CONUS.
Salida: results/tables/e1_counterfactuals_full.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config  # noqa: E402
from agrors.metrics import components  # noqa: E402

TR = ROOT / "data" / "processed" / "transitions"
RATIO = {
    "group_change": ("group_changed", "n"),
    "crop_gross": ("crop_turnover", "crop0"),
    "crop_type_change": ("crop_both_changed", "crop_both"),
    "nonag_change": ("nonag_changed", "nonag0"),
    "corn_corn": ("corn_corn", "corn0"),
    "cs_alternation": ("cs_alt", "cornsoy0"),
}
FIRST = 2014  # inicio de la línea base comparable (ver 03g)


def county_components() -> dict[int, pd.DataFrame]:
    out = {}
    for y0 in range(FIRST, 2025):
        c = components(pd.read_parquet(TR / f"{y0}_{y0 + 1}_nass30.parquet")).set_index("zone")
        c["crop_turnover"] = c.crop_loss + c.crop_gain
        c["cornsoy0"] = c.corn0 + c.soy0
        out[y0] = c
    return out


def estimates(rate: dict[int, float], same: list[int], switch: list[int]) -> dict[str, float]:
    brk, before, after = rate[2023], rate[2022], rate[2024]
    xs = np.array(same, dtype=float)
    slope, icpt = np.polyfit(xs, [rate[y] for y in same], 1)
    cf = {
        "adjacent": (before + after) / 2,
        "pre_break": before,
        "old_same_nlcd": float(np.mean([rate[y] for y in same])),
        "old_nlcd_switch": float(np.mean([rate[y] for y in switch])),
        "trend": slope * 2023 + icpt,
    }
    out = {f"break_minus_{k}": brk - v for k, v in cf.items()}
    out["persist_minus_pre_break"] = after - before
    out["persist_minus_old_nlcd_switch"] = after - cf["old_nlcd_switch"]
    # descomposición bajo el modelo (ver docstring)
    out["decomp_nlcd_nu"] = cf["old_nlcd_switch"] - cf["old_same_nlcd"]
    # ΔN = persist_minus_old_nlcd_switch
    out["decomp_inconsistency_I"] = (
        out["break_minus_old_nlcd_switch"] - out["persist_minus_old_nlcd_switch"] / 2
    )
    return out


def main() -> int:
    cfg = load_config()
    nlcd = {int(k): str(v) for k, v in cfg["nlcd_training"].items()}
    old = range(FIRST, 2023)  # pares 2014→15 … 2022→23
    same = [y for y in old if nlcd[y] == nlcd[y + 1]]
    switch = [y for y in old if nlcd[y] != nlcd[y + 1]]
    comp = county_components()
    zones = sorted(set.intersection(*(set(c.index) for c in comp.values())))
    rng = np.random.default_rng(cfg["seed"])
    reps = cfg["stats"]["bootstrap_reps"]
    idx = [rng.integers(0, len(zones), len(zones)) for _ in range(reps)]
    rows = []
    for metric, (num, den) in RATIO.items():
        arr = {y: c.loc[zones, [num, den]].to_numpy(dtype=float) for y, c in comp.items()}
        point = estimates({y: a[:, 0].sum() / a[:, 1].sum() * 100 for y, a in arr.items()}, same, switch)
        boot = [
            estimates({y: a[ix, 0].sum() / a[ix, 1].sum() * 100 for y, a in arr.items()}, same, switch)
            for ix in idx
        ]
        for k, v in point.items():
            b = np.array([d[k] for d in boot])
            rows.append(
                {
                    "metric": metric,
                    "estimand": k,
                    "estimate_pp": v,
                    "ci_low": np.quantile(b, 0.025),
                    "ci_high": np.quantile(b, 0.975),
                    "n_counties": len(zones),
                    "pairs_same_nlcd": " ".join(f"{y}-{y + 1}" for y in same),
                    "pairs_nlcd_switch": " ".join(f"{y}-{y + 1}" for y in switch),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "e1_counterfactuals_full.csv", index=False)
    print(out.pivot(index="estimand", columns="metric", values="estimate_pp").round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
