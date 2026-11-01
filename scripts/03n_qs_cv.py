"""Fase 3n: incertidumbre de las estimaciones de encuesta usadas en la Tabla 4.

Quick Stats publica, para una parte de las estimaciones de condado, el coeficiente de variación (CV, %).
Para la superficie sembrada de maíz, soja y algodón por condado en 2022–2025 (mismos data items que
agrors.quickstats.SHORT_DESC, dominio TOTAL, periodo anual) se resume la cobertura del CV y su mediana,
simple y ponderada por superficie.
Salida: results/tables/qs_cv_county.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402
from agrors.quickstats import SHORT_DESC, parse_value  # noqa: E402

RAW = ROOT / "data" / "raw" / "quickstats"
YEARS = range(2022, 2026)


def main() -> int:
    rows = []
    for crop in ("corn", "soybeans", "cotton"):
        descs = SHORT_DESC[(crop, "planted")]
        descs = descs if isinstance(descs, tuple) else (descs,)
        parts = []
        for y in YEARS:
            f = RAW / f"qs_county_{crop}_{y}.csv"
            if f.exists():
                parts.append(pd.read_csv(f, dtype=str))
        d = pd.concat(parts)
        d = d[d.short_desc.isin(descs) & (d.domain_desc == "TOTAL") & (d.reference_period_desc == "YEAR")]
        d = d[d.county_ansi.notna() & (d.county_ansi.str.strip() != "")]
        d["acres"] = parse_value(d["Value"])
        d["cv"] = pd.to_numeric(d["CV (%)"], errors="coerce")
        d = d[d.acres.notna()]
        for y, g in d.groupby("year"):
            has = g.cv.notna()
            rows.append(
                {
                    "crop": crop,
                    "year": int(y),
                    "n_estimates": len(g),
                    "share_with_cv": 100 * float(has.mean()),
                    "cv_median": float(g.cv.median()) if has.any() else np.nan,
                    "cv_area_weighted": float(np.average(g.cv[has], weights=g.acres[has]))
                    if has.any()
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "qs_cv_county.csv", index=False)
    print(out.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
