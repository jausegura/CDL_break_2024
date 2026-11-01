"""Fase 2d: Quick Stats en bruto → data/processed/quickstats_{county,state}.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402
from agrors.quickstats import tidy  # noqa: E402


def main() -> int:
    raw = ROOT / "data" / "raw" / "quickstats"
    for level in ("county", "state"):
        parts = []
        for f in sorted(raw.glob(f"qs_{level}_*.csv")):
            if f.stat().st_size == 0:
                continue
            parts.append(tidy(pd.read_csv(f, dtype=str), level))
        df = pd.concat(parts, ignore_index=True)
        out = ROOT / "data" / "processed" / f"quickstats_{level}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        cov = df.dropna(subset=["planted"]).groupby(["crop", "year"]).size().unstack("crop")
        cov.to_csv(ROOT / "results" / "tables" / f"quickstats_coverage_{level}.csv")
        print(level, len(df), "filas\n", cov.tail(6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
