"""Fase 3 (E3): consistencia de las Crop Sequence Boundaries entre ventanas que cruzan la ruptura.

(i)   Resumen por versión y estado: polígonos, superficie, mediana de tamaño.
(ii)  Cambio de etiqueta a nivel de polígono por par de años, en cada versión (ponderado por
      superficie): fracción que cambia de clase, maíz→maíz y alternancia maíz–soja.
(iii) Emparejamiento CSB 2016–23 ↔ CSB 2018–25 (punto interior de cada polígono 2018–25 dentro de
      un polígono 2016–23): acuerdo de etiquetas en los años comunes 2018–2023, polígonos sin
      pareja e IoU geométrico en una muestra aleatoria.
Salidas: results/tables/e3_csb_versions.csv, e3_csb_label_change.csv, e3_csb_match.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import CORN, SOYBEANS  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402

CSB = ROOT / "data" / "interim" / "csb"
TAB = ROOT / "results" / "tables"
IOU_BOOT = 1000  # réplicas bootstrap del IC de la mediana de IoU (muestra aleatoria simple)
IOU_SAMPLE = 5000


def load(window: str, state: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(CSB / f"csb{window}_{state}.parquet")


def label_change(g: pd.DataFrame, window: str, state: str) -> list[dict]:
    years = sorted(int(c[3:]) for c in g.columns if c.startswith("CDL"))
    w = g.CSBACRES.to_numpy()
    rows = []
    for y0, y1 in zip(years[:-1], years[1:], strict=True):
        a, b = g[f"CDL{y0}"].to_numpy(), g[f"CDL{y1}"].to_numpy()
        corn = a == CORN
        cs = (a == CORN) | (a == SOYBEANS)
        alt = ((a == CORN) & (b == SOYBEANS)) | ((a == SOYBEANS) & (b == CORN))
        rows.append(
            {
                "window": window,
                "state": state,
                "y0": y0,
                "y1": y1,
                "change_area_share": w[a != b].sum() / w.sum(),
                "change_polygon_share": float((a != b).mean()),
                "corn_corn": w[corn & (b == CORN)].sum() / w[corn].sum(),
                "cs_alternation": w[alt].sum() / w[cs].sum(),
                "corn_acres_y0": w[corn].sum(),
            }
        )
    return rows


def match(old: gpd.GeoDataFrame, new: gpd.GeoDataFrame, state: str, seed: int) -> dict:
    pts = gpd.GeoDataFrame(
        new[["CSBID", "CSBACRES"]].copy(),
        geometry=gpd.points_from_xy(new.INSIDE_X, new.INSIDE_Y),
        crs=new.crs,
    )
    if not np.isfinite(new.INSIDE_X).all():
        pts["geometry"] = new.geometry.representative_point()
    j = gpd.sjoin(pts, old[["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    matched = j["index_right"].notna().to_numpy()
    out = {
        "state": state,
        "n_new": len(new),
        "n_old": len(old),
        "share_new_unmatched": float(1 - matched.mean()),
        "share_new_unmatched_area": float(new.CSBACRES[~matched].sum() / new.CSBACRES.sum()),
    }
    oi = j.loc[matched, "index_right"].astype(int).to_numpy()
    ni = np.nonzero(matched)[0]
    w = new.CSBACRES.to_numpy()[ni]
    for y in range(2018, 2024):
        same = old[f"CDL{y}"].to_numpy()[oi] == new[f"CDL{y}"].to_numpy()[ni]
        out[f"label_agree_{y}"] = float(w[same].sum() / w.sum())
    # cuántos polígonos nuevos caen en el mismo polígono antiguo (fusión/división)
    cnt = np.bincount(oi, minlength=len(old))
    out["share_old_with_multiple_new"] = float((cnt > 1).sum() / max((cnt > 0).sum(), 1))
    # emparejamiento inverso (punto interior de cada polígono antiguo dentro de uno nuevo): asimetría
    rp = gpd.GeoDataFrame(geometry=old.geometry.representative_point(), crs=old.crs)
    jr = gpd.sjoin(rp, new[["geometry"]], how="left", predicate="within")
    jr = jr[~jr.index.duplicated()]
    out["share_old_unmatched"] = float(jr["index_right"].isna().mean())
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ni), size=min(IOU_SAMPLE, len(ni)), replace=False)
    gn = new.geometry.to_numpy()[ni[pick]]
    go = old.geometry.to_numpy()[oi[pick]]
    inter = np.array([a.intersection(b).area for a, b in zip(gn, go, strict=True)])
    union = np.array([a.union(b).area for a, b in zip(gn, go, strict=True)])
    iou = inter / union
    boot = [np.median(rng.choice(iou, size=len(iou))) for _ in range(IOU_BOOT)]
    out.update(
        {
            "iou_median": float(np.median(iou)),
            "iou_median_ci_low": float(np.quantile(boot, 0.025)),
            "iou_median_ci_high": float(np.quantile(boot, 0.975)),
            "iou_mean": float(iou.mean()),
            "iou_share_gt_0.9": float((iou > 0.9).mean()),
            "iou_n": int(len(iou)),
        }
    )
    return out


def main() -> int:
    cfg = load_config()
    states = cfg["pilot"]["states"]
    vers, lc, mt = [], [], []
    for state in states:
        gs = {}
        for window in cfg["csb"]["windows"]:
            p = CSB / f"csb{window}_{state}.parquet"
            if not p.exists():
                continue
            g = load(window, state)
            gs[window] = g
            vers.append(
                {
                    "window": window,
                    "state": state,
                    "n_polygons": len(g),
                    "total_acres": g.CSBACRES.sum(),
                    "median_acres": g.CSBACRES.median(),
                    "p10_acres": g.CSBACRES.quantile(0.1),
                    "p90_acres": g.CSBACRES.quantile(0.9),
                }
            )
            lc += label_change(g, window, state)
        if "2016-2023" in gs and "2018-2025" in gs:
            mt.append(match(gs["2016-2023"], gs["2018-2025"], state, cfg["seed"]))
            print(state, mt[-1], flush=True)
    pd.DataFrame(vers).to_csv(TAB / "e3_csb_versions.csv", index=False)
    pd.DataFrame(lc).to_csv(TAB / "e3_csb_label_change.csv", index=False)
    pd.DataFrame(mt).to_csv(TAB / "e3_csb_match.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
