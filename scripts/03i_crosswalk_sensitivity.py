"""Fase 3i: tabla de clases CDL — frecuencias de los códigos reasignados y sensibilidad de las cifras.

La primera versión del análisis (v1) asignaba 58 Clover/Wildflowers, 59 Sod/Grass Seed y 60 Switchgrass al
pastizal, 81 Clouds/No Data a "barren/other" y no asignaba 62 Pasture/Grass. La versión actual (cdl.GROUPS)
sigue la leyenda de USDA NASS: 58–60 son cultivos, 62 es pastizal y 81 es sin dato. Este script
documenta cuánto pesan esos códigos y cuánto cambian las cifras nacionales con una u otra tabla.

Salidas:
  results/tables/crosswalk_frequency.csv    % de píxeles válidos de cada código por año (CONUS, NASS 30 m)
  results/tables/crosswalk_sensitivity.csv  tasas por par y estimadores (exceso, salto, persistencia) con
                                            ambas tablas, por variante del mapa nuevo
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import GROUPS, group_lut  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.metrics import METRICS, components, excess, rates  # noqa: E402

TRANS = ROOT / "data" / "processed" / "transitions"
OUT = ROOT / "results" / "tables"
CODES = (58, 59, 60, 62, 81)

GROUPS_V1 = {k: set(v) for k, v in GROUPS.items()}
GROUPS_V1["crop"] -= {58, 59, 60}
GROUPS_V1["grass_pasture"] = {58, 59, 60, 176}
GROUPS_V1["barren_other"] |= {81}
LUTS = {"v1": group_lut(GROUPS_V1), "usda": group_lut()}


def national(path: Path, lut) -> pd.Series:
    t = pd.read_parquet(path, columns=["from", "to", "n"]).assign(zone=1)
    return rates(components(t, lut=lut)).iloc[0]


def main() -> int:
    y0, y1 = load_config()["full"]["years"]
    freq = []
    for y in range(y0, y1):
        t = pd.read_parquet(TRANS / f"{y}_{y + 1}_nass30.parquet", columns=["from", "to", "n"])
        tot = t.n.sum()
        years = [(y, "from")] + ([(y + 1, "to")] if y + 1 == y1 else [])
        for year, col in years:
            g = t.groupby(col).n.sum()
            for c in CODES:
                freq.append(
                    {
                        "year": year,
                        "code": c,
                        "pixels": int(g.get(c, 0)),
                        "share_pct": 100 * g.get(c, 0) / tot,
                    }
                )
    pd.DataFrame(freq).to_csv(OUT / "crosswalk_frequency.csv", index=False)

    rows = []
    for variant in ("nass30", "nn30", "maj30", "native10"):
        for a in range(y0, y1):
            name = variant if a + 1 >= 2024 else "nass30"
            p = TRANS / f"{a}_{a + 1}_{name}.parquet"
            if not p.exists():
                continue
            for xw, lut in LUTS.items():
                r = national(p, lut)
                rows.append(
                    {
                        "variant": variant,
                        "crosswalk": xw,
                        "y0": a,
                        "y1": a + 1,
                        **{m: 100 * r[m] for m in METRICS},
                    }
                )
    d = pd.DataFrame(rows)
    est = []
    for (variant, xw), g in d.groupby(["variant", "crosswalk"]):
        g = g.set_index("y1")
        if not {2023, 2024, 2025} <= set(g.index):
            continue
        for m in METRICS:
            before, brk, after = g.at[2023, m], g.at[2024, m], g.at[2025, m]
            est.append(
                {
                    "variant": variant,
                    "crosswalk": xw,
                    "metric": m,
                    "rate_2022_2023": before,
                    "rate_2023_2024": brk,
                    "rate_2024_2025": after,
                    "excess": excess(brk, before, after),
                    "step": brk - before,
                    "persist": after - before,
                    "max_abs_diff_any_pair": None,
                }
            )
    e = pd.DataFrame(est)
    # mayor diferencia absoluta (pp) entre tablas en cualquier par de la serie, por variante y métrica
    w = d.pivot_table(index=["variant", "y1"], columns="crosswalk", values=METRICS)
    for m in METRICS:
        diff = (w[(m, "usda")] - w[(m, "v1")]).abs().groupby(level="variant").max()
        sel = e.metric == m
        e.loc[sel, "max_abs_diff_any_pair"] = e.loc[sel, "variant"].map(diff)
    e.to_csv(OUT / "crosswalk_sensitivity.csv", index=False)
    nat = e[e.variant == "nass30"].pivot(index="metric", columns="crosswalk", values=["excess", "step"])
    print(nat.round(3).to_string())
    print(
        "máx. diferencia en cualquier par (pp):",
        e[e.variant == "nass30"].set_index("metric").max_abs_diff_any_pair.round(3).to_dict(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
