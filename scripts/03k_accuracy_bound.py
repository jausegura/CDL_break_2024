"""Fase 3k: qué permiten decir las matrices de error oficiales de NASS sobre el cambio falso entre años.

Fuente: matrices de error completas por estado (CDL_{2023,2024,2025}_accuracy_assessments.zip, enlazadas
desde el FAQ del CDL; filas = mapa, columnas = referencia), que incluyen pastizal y clases no agrícolas.
Por estado y año se agregan a los siete grupos de cobertura y se estima P_t(g'|g), la probabilidad de que
un píxel de referencia del grupo g aparezca en el mapa como g'. Si los errores de dos mapas fueran
independientes, la probabilidad de que un píxel que no cambia aparezca con grupo distinto sería
    D(t1, t2) = Σ_g w_g [1 − Σ_g' P_t1(g'|g) · P_t2(g'|g)]
con w_g la cuota del grupo g en el mapa 2023 del estado. Es una cota superior del cambio falso de grupo si
los errores de ambos años están positivamente correlacionados, e ignora los errores sistemáticos entre
sistemas de producción. Se calcula para 2023×2023 (dos mapas de la era antigua), 2023×2024 (cruza la
ruptura) y 2024×2025 (era nueva), y se compara con el cambio de grupo observado en 2022→23, 2023→24 y
2024→25. Nacional: media de los estados con matriz en los tres años ponderada por su superficie válida.
Los diseños de validación cambiaron en 2024 (muestra proporcional al tamaño del estado y a cultivo/no
cultivo), por lo que las matrices de ambas eras no son estrictamente comparables.
Salidas: results/tables/accuracy_bound.csv, accuracy_forage_classes.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.accuracy import group_matrix, state_matrices  # noqa: E402
from agrors.cdl import GROUP_ORDER, UNGROUPED, group_lut  # noqa: E402
from agrors.io import ROOT, download, load_config  # noqa: E402

TAB = ROOT / "results" / "tables"
RAW = ROOT / "data" / "raw" / "accuracy"
URL = "https://www.nass.usda.gov/Research_and_Science/Cropland/docs/CDL_{y}_accuracy_assessments.zip"
YEARS = (2023, 2024, 2025)
PAIRS = {"2022-2023": (2023, 2023), "2023-2024": (2023, 2024), "2024-2025": (2024, 2025)}
FORAGE = [36, 37, 176]


def conditional(g: np.ndarray) -> np.ndarray:
    """P(mapa g' | referencia g): columnas normalizadas (columnas sin referencia → NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return g / g.sum(axis=0, keepdims=True)


def bound(p1: np.ndarray, p2: np.ndarray, w: pd.Series) -> float:
    agree = np.nansum(p1 * p2, axis=0)  # por grupo de referencia
    ok = ~np.isnan(p1).all(axis=0) & ~np.isnan(p2).all(axis=0)
    ww = w.to_numpy() * ok
    return float((ww * (1 - agree)).sum() / ww.sum())


def main() -> int:
    pilot = load_config()["pilot"]["states"]
    lut = group_lut()
    ng = len(GROUP_ORDER)
    for y in YEARS:
        download(
            URL.format(y=y),
            RAW / f"CDL_{y}_accuracy_assessments.zip",
            "Public domain (USDA NASS)",
            f"Matrices de error CDL {y} (FAQ NASS)",
        )
    mats = {y: state_matrices(RAW / f"CDL_{y}_accuracy_assessments.zip", y) for y in YEARS}
    states = sorted(set.intersection(*(set(m) for m in mats.values())) - {"NATIONAL"})
    cond = {y: {s: conditional(group_matrix(mats[y][s], lut, ng, UNGROUPED)) for s in states} for y in YEARS}
    flows = pd.read_csv(TAB / "e1_group_flows_full.csv")
    f23 = flows[(flows.y0 == 2023) & (flows.variant == "nass30")]
    agg = pd.read_csv(TAB / "e1_aggregate_full.csv")
    agg = agg[agg.variant == "nass30"].set_index(["STUSPS", "y0"])
    rows = []
    for s in states:
        w = f23[f23.region == s].groupby("group_from").share.sum().reindex(GROUP_ORDER).fillna(0)
        for pair, (t1, t2) in PAIRS.items():
            y0 = int(pair[:4])
            rows.append(
                {
                    "state": s,
                    "pair": pair,
                    "matrices": f"{t1}x{t2}",
                    "bound_pct": 100 * bound(cond[t1][s], cond[t2][s], w),
                    "observed_pct": 100 * float(agg.loc[(s, y0), "group_change"]),
                    "area_weight": float(agg.loc[(s, 2022), "n"]),
                }
            )
    d = pd.DataFrame(rows)
    nat = (
        d.groupby("pair")
        .apply(
            lambda g: pd.Series(
                {
                    "bound_pct": np.average(g.bound_pct, weights=g.area_weight),
                    "observed_pct": np.average(g.observed_pct, weights=g.area_weight),
                    "n_states": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .assign(state="CONUS (states with matrices)")
    )
    out = pd.concat([nat, d[d.state.isin(pilot)]], ignore_index=True)
    out.to_csv(TAB / "accuracy_bound.csv", index=False)
    # exactitud del productor de heno y pastizal por estado piloto y año, de las matrices completas
    fr = []
    for y in YEARS:
        for s in pilot:
            m = mats[y].get(s)
            if m is None:
                continue
            for c in FORAGE:
                ref = m[:, c].sum()
                fr.append(
                    {
                        "scope": s,
                        "year": y,
                        "code": c,
                        "ref_pixels": int(ref),
                        "producer_acc": 100 * m[c, c] / ref if ref else np.nan,
                    }
                )
    pd.DataFrame(fr).to_csv(TAB / "accuracy_forage_classes.csv", index=False)
    print(out.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
