"""Fase 3 (E1, sensibilidad): rotación cultivo↔no cultivo con tres definiciones de cultivo.

  default  cdl.GROUPS["crop"] (leyenda USDA): incluye heno/alfalfa (36, 37), cultivos herbáceos forrajeros o
           de pradera (58 Clover/Wildflowers, 59 Sod/Grass Seed, 60 Switchgrass) y barbecho (61)
  no_hay   excluye 36, 37 y 58–60, las clases que más se confunden con el pastizal
  strict   excluye además el barbecho (61)
Las clases excluidas pasan a "no cultivo". Serie nacional nass30, todos los pares.
Salidas:
  results/tables/e1_cropdef_sensitivity_full.csv  tasas nacionales por par y definición, cambio neto en
                                                  millones de acres y flujo cultivo→pastizal (% de píxeles)
  results/tables/e1_cropdef_estimators_full.csv   recambio bruto y cambio neto en los tres pares de la
                                                  ruptura con exceso, salto y persistencia (IC 95 % por
                                                  bootstrap de condados, percentiles) y cuota de condados
                                                  con exceso positivo (denominador ≥ stats.county_min_den)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import GROUPS, UNGROUPED, group_lut  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.quickstats import ACRE_M2  # noqa: E402

TR = ROOT / "data" / "processed" / "transitions"
FORAGE = {36, 37, 58, 59, 60}
DEFS = {
    "default": GROUPS["crop"],
    "no_hay": GROUPS["crop"] - FORAGE,
    "strict": GROUPS["crop"] - FORAGE - {61},
}


def crop_masks(t: pd.DataFrame, codes: set[int]) -> tuple[np.ndarray, np.ndarray]:
    is_crop = np.zeros(256, dtype=bool)
    is_crop[list(codes)] = True
    return is_crop[t["from"].to_numpy()], is_crop[t["to"].to_numpy()]


def county_components(t: pd.DataFrame, codes: set[int]) -> pd.DataFrame:
    """crop0, crop1, pérdida y ganancia de cultivo por condado (píxeles)."""
    a, b = crop_masks(t, codes)
    n = t.n.to_numpy(dtype=float)
    c = pd.DataFrame({"zone": t.zone.to_numpy(), "crop0": n * a, "crop1": n * b, "turn": n * (a != b)})
    return c.groupby("zone").sum()


def estimators(comp: dict[int, pd.DataFrame], cfg: dict) -> list[dict]:
    """Exceso, salto y persistencia del recambio bruto y del cambio neto, con bootstrap de condados."""
    zones = sorted(set.intersection(*(set(c.index) for c in comp.values())))
    arr = {y: comp[y].loc[zones].to_numpy() for y in comp}  # columnas crop0, crop1, turn
    rng = np.random.default_rng(cfg["seed"])
    idx = [rng.integers(0, len(zones), len(zones)) for _ in range(cfg["stats"]["bootstrap_reps"])]

    def rates(ix):
        out = {}
        for y, a in arr.items():
            s0, s1, tu = a[ix, 0].sum(), a[ix, 1].sum(), a[ix, 2].sum()
            out[y] = {"crop_gross": 100 * tu / s0, "crop_net": 100 * (s1 - s0) / s0}
        return out

    def est(r, m):
        b, k, a = r[2022][m], r[2023][m], r[2024][m]
        return {"excess": k - (b + a) / 2, "step": k - b, "persist": a - b}

    point = rates(np.arange(len(zones)))
    boots = [rates(ix) for ix in idx]
    rows = []
    for m in ("crop_gross", "crop_net"):
        e = est(point, m)
        be = [est(r, m) for r in boots]
        row = {"metric": m, **{f"rate_{y}_{y + 1}": point[y][m] for y in (2022, 2023, 2024)}}
        for k, v in e.items():
            vals = np.array([d[k] for d in be])
            row |= {k: v, f"{k}_ci_low": np.quantile(vals, 0.025), f"{k}_ci_high": np.quantile(vals, 0.975)}
        # cuota de condados con exceso positivo
        den = np.all([arr[y][:, 0] >= cfg["stats"]["county_min_den"] for y in arr], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = {
                y: (a[:, 2] / a[:, 0] if m == "crop_gross" else (a[:, 1] - a[:, 0]) / a[:, 0])
                for y, a in arr.items()
            }
        ex = (rc[2023] - (rc[2022] + rc[2024]) / 2)[den]
        row |= {"share_counties_positive": 100 * float((ex > 0).mean()), "n_counties": int(den.sum())}
        rows.append(row)
    return rows


def main() -> int:
    cfg = load_config()
    lut = group_lut()
    grass = np.isin(np.arange(256), list(GROUPS["grass_pasture"]))
    rows, est_rows = [], []
    comps: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in DEFS}
    for f in sorted(TR.glob("*_nass30.parquet")):
        y0, y1, _ = f.stem.split("_")
        t = pd.read_parquet(f)
        t = t[(lut[t["from"].to_numpy()] != UNGROUPED) & (lut[t["to"].to_numpy()] != UNGROUPED)]
        n = t.n.to_numpy(dtype=float)
        to_grass = grass[t["to"].to_numpy()]
        for name, codes in DEFS.items():
            a, b = crop_masks(t, codes)
            c0, c1 = n[a].sum(), n[b].sum()
            rows.append(
                {
                    "y0": int(y0),
                    "y1": int(y1),
                    "definition": name,
                    "crop_gross": (n[a & ~b].sum() + n[~a & b].sum()) / c0,
                    "crop_net": (c1 - c0) / c0,
                    "crop_loss": n[a & ~b].sum() / c0,
                    "crop_gain": n[~a & b].sum() / c0,
                    "crop_net_macres": (c1 - c0) * 900 / ACRE_M2 / 1e6,
                    "crop_to_grass_share": n[a & to_grass].sum() / n.sum(),
                }
            )
            if int(y0) in (2022, 2023, 2024):
                comps[name][int(y0)] = county_components(t, codes)
    for name, comp in comps.items():
        est_rows += [{"definition": name, **r} for r in estimators(comp, cfg)]
    out = pd.DataFrame(rows).sort_values(["definition", "y0"])
    out.to_csv(ROOT / "results" / "tables" / "e1_cropdef_sensitivity_full.csv", index=False)
    est = pd.DataFrame(est_rows)
    est.to_csv(ROOT / "results" / "tables" / "e1_cropdef_estimators_full.csv", index=False)
    print(out[out.y0 >= 2021].round(4).to_string(index=False))
    print(est.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
