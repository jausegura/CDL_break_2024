"""Fase 3 (E5): reglas de armonización y cuánto reducen el exceso de cambio (píxeles de parcela CSB).

Reglas (sobre los estratos de E2, piloto):
  R0 ninguna: producto NASS 30 m, todos los píxeles de parcela
  R1 mayoría 3x3 desde 10 m para 2024–25
  R2 máscara de borde: solo píxeles a ≥30 m del borde CSB
  R3 máscara de borde estricta: ≥60 m
  R4 = R1 + R2;  R5 = R1 + R3
Para cada regla y métrica (cambio de clase; cultivo↔no cultivo) se dan las tasas por par, el
exceso, el salto (step) y la reducción relativa del step frente a R0.
Salida: results/tables/e5_harmonization.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config  # noqa: E402

TAB = ROOT / "results" / "tables"
RULES = {
    "R0_none": ("nass30", (0, 1, 2)),
    "R1_majority": ("maj30", (0, 1, 2)),
    "R2_edge30": ("nass30", (1, 2)),
    "R3_edge60": ("nass30", (2,)),
    "R4_majority_edge30": ("maj30", (1, 2)),
    "R5_majority_edge60": ("maj30", (2,)),
}


def main() -> int:
    d = pd.read_csv(TAB / "e2_strata_pilot.csv")
    # solo estados del piloto (los polígonos fronterizos caen a veces en condados vecinos)
    d = d[d.state.isin(load_config()["pilot"]["states"])]
    rows = []
    for region in ["ALL", *sorted(d.state.unique())]:
        dd = d if region == "ALL" else d[d.state == region]
        for rule, (variant, bins) in RULES.items():
            rates = {}
            for y0, y1 in [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]:
                v = variant if y1 >= 2024 else "nass30"
                s = dd[(dd.y0 == y0) & (dd.y1 == y1) & (dd.variant == v) & dd.edge_bin.isin(bins)]
                if s.empty:
                    continue
                rates[(y0, y1)] = {
                    "class_change": s.changed.sum() / s.n.sum(),
                    "crop_noncrop": s.crop_noncrop_changed.sum() / s.n.sum(),
                    "pixels": s.n.sum(),
                }
            if (2023, 2024) not in rates or (2024, 2025) not in rates:
                continue
            for m in ("class_change", "crop_noncrop"):
                b, k, a = rates[(2022, 2023)][m], rates[(2023, 2024)][m], rates[(2024, 2025)][m]
                rows.append(
                    {
                        "region": region,
                        "rule": rule,
                        "metric": m,
                        "rate_2021_22": rates[(2021, 2022)][m],
                        "rate_2022_23": b,
                        "rate_2023_24": k,
                        "rate_2024_25": a,
                        "excess": k - (b + a) / 2,
                        "step": k - b,
                        "persist": a - b,
                        "pixels_2023_24": int(rates[(2023, 2024)]["pixels"]),
                    }
                )
    out = pd.DataFrame(rows)
    base = out[out.rule == "R0_none"].set_index(["region", "metric"])
    for c in ("step", "persist", "excess"):
        ref = base[c].reindex(pd.MultiIndex.from_frame(out[["region", "metric"]])).to_numpy()
        out[f"{c}_reduction_vs_R0"] = 1 - out[c] / ref
    out.to_csv(TAB / "e5_harmonization.csv", index=False)
    print(out[out.region == "ALL"].round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
