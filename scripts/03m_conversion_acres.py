"""Fase 3m: consecuencia aguas abajo — la conversión pastizal↔cultivo que registraría una evaluación que
diferencie mapas CDL de años consecutivos (producto NASS 30 m, CONUS).

Para cada par se calculan los acres que pasan de pastizal/pradera a cultivo y de cultivo a pastizal
(cuota de píxeles válidos × píxeles válidos × 900 m² / acre) y se comparan con la media de los pares
2014→15 … 2022→23 (era antigua). El exceso de 2023→24 (cultivo→pastizal) y de 2024→25 (pastizal→cultivo,
"reconversión") es la conversión aparente atribuible a la ruptura bajo el contrafactual de la media antigua.
Salida: results/tables/conversion_acres_full.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402
from agrors.quickstats import ACRE_M2  # noqa: E402

TAB = ROOT / "results" / "tables"
FLOWS = {"grass_to_crop": ("grass_pasture", "crop"), "crop_to_grass": ("crop", "grass_pasture")}
OLD = range(2014, 2023)


def main() -> int:
    f = pd.read_csv(TAB / "e1_group_flows_full.csv")
    f = f[(f.region == "ALL") & (f.variant == "nass30")]
    a = pd.read_csv(TAB / "e1_aggregate_full.csv")
    n = a[(a.STUSPS == "ALL") & (a.variant == "nass30")].set_index("y0").n
    rows = []
    for y0, g in f.groupby("y0"):
        row = {"y0": int(y0), "y1": int(y0) + 1}
        for k, (g0, g1) in FLOWS.items():
            share = g[(g.group_from == g0) & (g.group_to == g1)].share.sum()
            row[f"{k}_macres"] = share * n[y0] * 900 / ACRE_M2 / 1e6
        rows.append(row)
    d = pd.DataFrame(rows).sort_values("y0")
    for k in FLOWS:
        col = f"{k}_macres"
        base = d[d.y0.isin(OLD)][col].mean()
        d[f"{k}_old_mean_macres"] = base
        d[f"{k}_excess_macres"] = d[col] - base
    d.to_csv(TAB / "conversion_acres_full.csv", index=False)
    print(d[d.y0 >= 2020].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
