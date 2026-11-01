"""Fase 3 (E1-bis): separa la ruptura de método de los cambios de edición NLCD.

Clasifica cada par de años consecutivos (serie nacional nass30) en un régimen:
  old_same_nlcd    era antigua, misma edición NLCD en ambos años
  old_nlcd_switch  era antigua, cambia la edición NLCD (2009→10, 2010→11, 2013→14, 2018→19, 2020→21)
  break_2023_24    cambio de método (See5 30 m → RF 10 m) + cambio NLCD 2019→2021
  new_2024_25      dentro de la era nueva + cambio NLCD 2021→2024
y resume las métricas de E1 por régimen (media y rango), para toda la serie y para 2014–2025 (la
CDL de 2008–2013 es más ruidosa y no es una línea base comparable).
Salidas: results/tables/e1_pairs_regime_full.csv, e1_regime_summary_full.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config  # noqa: E402

TAB = ROOT / "results" / "tables"
METRICS = [
    "group_change",
    "crop_gross",
    "nonag_change",
    "crop_type_change",
    "any_change",
    "corn_corn",
    "cs_alternation",
]


def regime(y0: int, y1: int, nlcd: dict) -> str:
    if y0 == 2023:
        return "break_2023_24"
    if y0 >= 2024:
        return "new_2024_25"
    return "old_nlcd_switch" if nlcd[y0] != nlcd[y1] else "old_same_nlcd"


def main() -> int:
    cfg = load_config()
    nlcd = {int(k): str(v) for k, v in cfg["nlcd_training"].items()}
    a = pd.read_csv(TAB / "e1_aggregate_full.csv")
    a = a[(a.STUSPS == "ALL") & (a.variant == "nass30")].sort_values("y0").copy()
    a["regime"] = [regime(y0, y1, nlcd) for y0, y1 in zip(a.y0, a.y1, strict=True)]
    a["nlcd_from"] = a.y0.map(nlcd)
    a["nlcd_to"] = a.y1.map(nlcd)
    a[["y0", "y1", "regime", "nlcd_from", "nlcd_to", *METRICS]].to_csv(
        TAB / "e1_pairs_regime_full.csv", index=False
    )
    rows = []
    for period, sub in [("2008-2025", a), ("2014-2025", a[a.y0 >= 2014])]:
        for reg, g in sub.groupby("regime"):
            r = {
                "period": period,
                "regime": reg,
                "n_pairs": len(g),
                "pairs": ";".join(f"{x}-{y}" for x, y in zip(g.y0, g.y1, strict=True)),
            }
            for m in METRICS:
                r[f"{m}_mean"], r[f"{m}_min"], r[f"{m}_max"] = g[m].mean(), g[m].min(), g[m].max()
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "e1_regime_summary_full.csv", index=False)
    print(
        a[["y0", "y1", "regime", "group_change", "crop_gross", "nonag_change", "crop_type_change"]]
        .round(4)
        .to_string(index=False)
    )
    print(
        out[["period", "regime", "n_pairs", *[c for c in out.columns if c.endswith("_mean")]]]
        .round(4)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
