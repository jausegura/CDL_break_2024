"""Fase 4: tablas finales en Markdown (results/tables/*.md) a partir de los CSV de resultados.

Uso: python scripts/04b_tables.py [pilot|full]  (escala de E1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402

TAB = ROOT / "results" / "tables"


def md(df: pd.DataFrame, path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


def pct(x):
    return np.round(100 * x, 2)


def table_e1(scope: str) -> None:
    ex = pd.read_csv(TAB / f"e1_excess_{scope}.csv")
    ex = ex[ex.region == "ALL"]
    out = pd.DataFrame(
        {
            "variant": ex.variant,
            "metric": ex.metric,
            "2022→23 (%)": pct(ex.rate_2022_23),
            "2023→24 (%)": pct(ex.rate_2023_24),
            "2024→25 (%)": pct(ex.rate_2024_25),
            "excess (pp) [95% CI]": [
                f"{pct(a)} [{pct(b)}, {pct(c)}]"
                for a, b, c in zip(ex.excess, ex.ci_low, ex.ci_high, strict=True)
            ],
            "step (pp) [95% CI]": [
                f"{pct(a)} [{pct(b)}, {pct(c)}]"
                for a, b, c in zip(ex.step, ex.step_ci_low, ex.step_ci_high, strict=True)
            ],
            "persist (pp) [95% CI]": [
                f"{pct(a)} [{pct(b)}, {pct(c)}]"
                for a, b, c in zip(ex.persist, ex.persist_ci_low, ex.persist_ci_high, strict=True)
            ],
            "counties": ex.n_counties,
            "share counties excess>0": np.round(ex.share_counties_positive, 3),
            "Wilcoxon p (vs 22→23)": ex["wilcoxon_p_vs_2022_23"].map(lambda p: f"{p:.1e}"),
        }
    )
    md(out, TAB / f"table_e1_excess_{scope}.md", f"E1 — change rates and excess at the 2024 break ({scope})")


def table_qs(scope: str) -> None:
    d = pd.read_csv(TAB / f"e1_qs_did_{scope}.csv")
    cols = [c for c in d.columns if c.startswith("ratio_cdl_qs_")]
    out = d[["level", "crop", "variant", "n_counties", *cols]].copy()
    out[cols] = out[cols].round(3)
    out["excess Δlog(CDL/QS) [95% CI]"] = [
        f"{a:.3f} [{b:.3f}, {c:.3f}]"
        for a, b, c in zip(d.excess_dlog_ratio, d.ci_low, d.ci_high, strict=True)
    ]
    out = out.rename(columns={"n_counties": "n units"})
    md(out, TAB / f"table_e1_quickstats_{scope}.md", f"E1 — CDL area vs NASS planted area ({scope})")


def table_e2() -> None:
    p = TAB / "e2_strata_pilot.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    rows = []
    for (y0, y1, v), g in d.groupby(["y0", "y1", "variant"]):
        for eb in (0, 1, 2):
            s = g[g.edge_bin == eb]
            rows.append(
                {
                    "pair": f"{y0}→{y1}",
                    "variant": v,
                    "edge": ["<30 m", "30–60 m", "≥60 m"][eb],
                    "pixels": int(s.n.sum()),
                    "class change (%)": pct(s.changed.sum() / s.n.sum()),
                    "crop↔non-crop (%)": pct(s.crop_noncrop_changed.sum() / s.n.sum()),
                }
            )
    md(pd.DataFrame(rows), TAB / "table_e2_edge.md", "E2 — change by distance to CSB field edge (pilot)")


def table_e4() -> None:
    p = TAB / "e4_metrics.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    f1 = [c for c in d.columns if c.startswith("f1_") and c != "f1_macro"]
    g = d.groupby(["train", "test"])[["oa", "f1_macro", *f1]].agg(["mean", "std"])
    out = pd.DataFrame({"train": [i[0] for i in g.index], "test": [i[1] for i in g.index]})
    for c in ["oa", "f1_macro", *f1]:
        out[c] = [
            f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else ""
            for m, s in zip(g[(c, "mean")], g[(c, "std")].fillna(0), strict=True)
        ]
    md(out, TAB / "table_e4_models.md", "E4 — accuracy by label era (mean ± sd over seeds)")


def table_data() -> None:
    rows = [
        ("CDL 2008–2023 (30 m, See5)", "USDA NASS national GeoTIFF zips", "Public domain", "E1–E5"),
        (
            "CDL 2024–2025 (30 m nearest-neighbour resample)",
            "USDA NASS national GeoTIFF zips",
            "Public domain",
            "E1–E5",
        ),
        (
            "CDL 2024–2025 (10 m, random forest)",
            "Source Cooperative Icechunk copy of NASS rasters (pilot "
            "states; verified against official files)",
            "Public domain",
            "E1, E2, E4, E5",
        ),
        (
            "Crop Sequence Boundaries 2016–23, 2017–24, 2018–25 (rev23)",
            "USDA NASS",
            "Public domain",
            "E2, E3",
        ),
        ("NASS Quick Stats (planted/harvested area)", "USDA NASS Quick Stats API", "Public domain", "E1"),
        ("County boundaries 2023 (1:500k)", "US Census Bureau", "Public domain", "E1"),
        ("HLS v2.0 L30/S30", "NASA LP DAAC via Microsoft Planetary Computer", "NASA open data", "E4"),
        ("CropSight-US v1.0.1", "Zenodo 10.5281/zenodo.19501943", "CC BY 4.0", "E4"),
        ("CDL accuracy reports (metadata)", "USDA NASS", "Public domain", "Context"),
    ]
    md(
        pd.DataFrame(rows, columns=["Dataset", "Source", "Licence", "Used in"]),
        TAB / "table_data.md",
        "Data sources and licences",
    )


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    table_e1(scope)
    table_qs(scope)
    table_e2()
    table_e4()
    table_data()
    print("tablas markdown escritas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
