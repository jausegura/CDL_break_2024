"""Fase 3l (E2): separa los efectos de borde y de tamaño de parcela, que están confundidos (las parcelas
pequeñas tienen más píxeles de borde).

Datos: results/tables/e2_strata_pilot.csv (producto NASS 30 m; celdas estado × decil de tamaño × banda de
distancia al borde × clase de partida). Resultado: salto del cambio cultivo↔no cultivo,
rate(2023→24) − rate(2022→23), en cada celda.
  1. Tabla cruzada banda de borde × decil (seis estados agregados): tasas y salto.
  2. Regresión lineal ponderada (peso = píxeles de la celda en 2023→24) del salto de cada celda sobre
     indicadores de banda de borde y de decil de tamaño, con efectos fijos de estado y de clase de partida.
     Se informa cada coeficiente en el modelo conjunto y en los modelos con solo borde o solo tamaño, y el
     rango de los coeficientes al excluir cada estado (no hay inferencia por píxel: los píxeles de una
     parcela no son independientes).
Salidas: results/tables/e2_edge_size_cross.csv, e2_edge_size_model.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config  # noqa: E402

TAB = ROOT / "results" / "tables"
EDGE = {0: "<30 m", 1: "30-60 m", 2: ">=60 m"}


def cells(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["state", "size_decile", "edge_bin", "from_class"]
    w = d.pivot_table(index=keys, columns="y0", values=["n", "crop_noncrop_changed"], aggfunc="sum").dropna()
    out = pd.DataFrame(index=w.index)
    for y in (2022, 2023):
        out[f"rate_{y}"] = 100 * w[("crop_noncrop_changed", y)] / w[("n", y)]
    out["step"] = out.rate_2023 - out.rate_2022
    out["weight"] = w[("n", 2023)]
    return out[out.weight > 0].reset_index()


def design(c: pd.DataFrame, edge: bool, size: bool) -> tuple[np.ndarray, list[str]]:
    cols, names = [np.ones(len(c))], ["intercept"]
    if edge:
        for b in (0, 1):  # referencia: interior ≥60 m
            cols.append((c.edge_bin == b).to_numpy(float))
            names.append(f"edge_{EDGE[b]}")
    if size:
        for k in range(9):  # referencia: decil 10 (parcelas mayores)
            cols.append((c.size_decile == k).to_numpy(float))
            names.append(f"decile_{k + 1}")
    for col in ("state", "from_class"):
        levels = sorted(c[col].unique())
        for lv in levels[1:]:
            cols.append((c[col] == lv).to_numpy(float))
            names.append(f"{col}_{lv}")
    return np.column_stack(cols), names


def wls(c: pd.DataFrame, edge: bool, size: bool) -> dict[str, float]:
    x, names = design(c, edge, size)
    sw = np.sqrt(c.weight.to_numpy())
    beta, *_ = np.linalg.lstsq(x * sw[:, None], c.step.to_numpy() * sw, rcond=None)
    return dict(zip(names, beta, strict=True))


def main() -> int:
    states = load_config()["pilot"]["states"]
    d = pd.read_csv(TAB / "e2_strata_pilot.csv")
    d = d[d.state.isin(states) & (d.variant == "nass30") & d.y0.isin([2022, 2023])]
    g = d.groupby(["edge_bin", "size_decile", "y0"])[["n", "crop_noncrop_changed"]].sum().reset_index()
    g["rate"] = 100 * g.crop_noncrop_changed / g.n
    cross = g.pivot_table(index=["edge_bin", "size_decile"], columns="y0", values="rate").reset_index()
    cross.columns = ["edge_bin", "size_decile", "rate_2022_2023", "rate_2023_2024"]
    cross["step"] = cross.rate_2023_2024 - cross.rate_2022_2023
    share = g[g.y0 == 2023].set_index(["edge_bin", "size_decile"]).n
    cross["share_of_pixels"] = (
        100
        * share.reindex(pd.MultiIndex.from_frame(cross[["edge_bin", "size_decile"]])).to_numpy()
        / share.sum()
    )
    cross["edge"] = cross.edge_bin.map(EDGE)
    cross["size_decile"] += 1
    cross.to_csv(TAB / "e2_edge_size_cross.csv", index=False)
    c = cells(d)
    rows = []
    for model, (e, s) in {
        "joint": (True, True),
        "edge_only": (True, False),
        "size_only": (False, True),
    }.items():
        full = wls(c, e, s)
        loo = [wls(c[c.state != st], e, s) for st in states]
        for k, v in full.items():
            if k.startswith(("edge_", "decile_")):
                vals = [m[k] for m in loo]
                rows.append(
                    {"model": model, "term": k, "coef_pp": v, "loo_min": min(vals), "loo_max": max(vals)}
                )
    m = pd.DataFrame(rows)
    m.to_csv(TAB / "e2_edge_size_model.csv", index=False)
    print(cross.pivot(index="size_decile", columns="edge", values="step").round(1).to_string())
    print(
        m[m.term.isin(["edge_<30 m", "edge_30-60 m", "decile_1", "decile_5"])].round(2).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
