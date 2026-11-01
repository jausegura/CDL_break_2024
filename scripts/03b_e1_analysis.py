"""Fase 3 (E1, E3-píxel): métricas por condado, exceso en la ruptura, placebo y contraste Quick Stats.

Entradas: data/processed/transitions/*.parquet (03a), counties.csv, quickstats_county.parquet.
Salidas (results/tables/, sufijo _{scope}):
  e1_county_metrics       métricas por condado, par y variante
  e1_aggregate            métricas agregadas (suma de componentes) por estado y total
  e1_excess               exceso en 2023→24 por métrica/variante/región con IC bootstrap y Wilcoxon
  e1_placebo              exceso "falso" en cada año de la era antigua (solo scope full)
  e1_area_county          área CDL por condado/cultivo/año/variante y Quick Stats (acres)
  e1_qs_did               exceso de Δlog(CDL/QS) en 2023→24 con IC bootstrap
Uso: python scripts/03b_e1_analysis.py [pilot|full]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import AREA_CODES, AREA_CODES_SINGLE  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.metrics import METRICS, bootstrap_ratio_stats, components, rates  # noqa: E402
from agrors.quickstats import ACRE_M2  # noqa: E402

TR = ROOT / "data" / "processed" / "transitions"
TAB = ROOT / "results" / "tables"
# numerador y denominador de cada métrica (para agregar y para el bootstrap)
RATIO = {
    "any_change": ("changed", "n"),
    "group_change": ("group_changed", "n"),
    "crop_gross": ("crop_turnover", "crop0"),
    "crop_type_change": ("crop_both_changed", "crop_both"),
    "nonag_change": ("nonag_changed", "nonag0"),
    "corn_corn": ("corn_corn", "corn0"),
    "cs_alternation": ("cs_alt", "cornsoy0"),
}
PIX_M2 = {"nass30": 900.0, "maj30": 900.0, "native10": 100.0, "nn30": 900.0}


def load_transitions(scope: str) -> pd.DataFrame:
    suffix = "_pilot" if scope == "pilot" else ""
    rows = []
    for f in sorted(TR.glob(f"*{suffix}.parquet")):
        parts = f.stem.replace("_pilot", "").split("_")
        if scope == "full" and f.stem.endswith("_pilot"):
            continue
        y0, y1, variant = int(parts[0]), int(parts[1]), parts[2]
        c = components(pd.read_parquet(f))
        c["y0"], c["y1"], c["variant"] = y0, y1, variant
        c["px_m2"] = PIX_M2[variant]
        rows.append(c)
    df = pd.concat(rows, ignore_index=True)
    df["crop_turnover"] = df.crop_loss + df.crop_gain
    df["cornsoy0"] = df.corn0 + df.soy0
    return df


def with_old_era_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Los pares de la era antigua son iguales en todas las variantes: se replican."""
    old = df[(df.y1 <= 2023) & (df.variant == "nass30")]
    extra = [old.assign(variant=v) for v in ("maj30", "native10", "nn30")]  # px_m2 sigue siendo 900
    return pd.concat([df, *extra], ignore_index=True)


def aggregate(comp: pd.DataFrame, counties: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in comp.columns if c not in ("zone", "y0", "y1", "variant", "px_m2")]
    c = comp.merge(counties[["zone", "STUSPS"]], on="zone")
    by_state = c.groupby(["STUSPS", "y0", "y1", "variant"], as_index=False)[num_cols].sum()
    total = c.groupby(["y0", "y1", "variant"], as_index=False)[num_cols].sum().assign(STUSPS="ALL")
    out = rates(pd.concat([by_state, total], ignore_index=True))
    out["crop_gross"] = out.crop_turnover / out.crop0
    return out


def excess_table(comp: pd.DataFrame, counties: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    reps, seed = cfg["stats"]["bootstrap_reps"], cfg["seed"]
    c = comp.merge(counties[["zone", "STUSPS"]], on="zone")
    rows = []
    regions = ["ALL", *sorted(c.STUSPS.unique())]
    for variant in ("nass30", "maj30", "native10", "nn30"):
        sel = {
            "before": c[(c.y0 == 2022) & (c.y1 == 2023) & (c.variant == "nass30")],
            "break": c[(c.y0 == 2023) & (c.y1 == 2024) & (c.variant == variant)],
            "after": c[(c.y0 == 2024) & (c.y1 == 2025) & (c.variant == variant)],
        }
        if any(v.empty for v in sel.values()):
            continue
        for region in regions:
            sub = {
                k: (v if region == "ALL" else v[v.STUSPS == region]).set_index("zone") for k, v in sel.items()
            }
            common = sub["before"].index.intersection(sub["break"].index).intersection(sub["after"].index)
            for metric, (num, den) in RATIO.items():
                agg = {k: sub[k].loc[common, num].sum() / sub[k].loc[common, den].sum() for k in sub}
                est = agg["break"] - (agg["before"] + agg["after"]) / 2
                boot = bootstrap_ratio_stats({k: sub[k] for k in sub}, num, den, reps, seed)
                boot_st = (
                    bootstrap_ratio_stats(
                        {k: sub[k] for k in sub}, num, den, reps, seed, clusters=sub["before"].STUSPS
                    )
                    if region == "ALL"
                    else None
                )
                # nivel condado (condados con denominador > 0 en los tres pares)
                rc = {k: sub[k].loc[common, num] / sub[k].loc[common, den] for k in sub}
                ok = np.isfinite(rc["before"]) & np.isfinite(rc["break"]) & np.isfinite(rc["after"])
                ok &= county_den_ok(sub, common, den, cfg["stats"]["county_min_den"])
                ex_c = rc["break"][ok] - (rc["before"][ok] + rc["after"][ok]) / 2
                p_before = p_after = np.nan
                if ok.sum() >= 10:
                    p_before = wilcoxon(rc["break"][ok], rc["before"][ok]).pvalue
                    p_after = wilcoxon(rc["break"][ok], rc["after"][ok]).pvalue
                rows.append(
                    {
                        "region": region,
                        "variant": variant,
                        "metric": metric,
                        "rate_2022_23": agg["before"],
                        "rate_2023_24": agg["break"],
                        "rate_2024_25": agg["after"],
                        "excess": est,
                        "ci_low": np.quantile(boot["excess"], 0.025),
                        "ci_high": np.quantile(boot["excess"], 0.975),
                        "step": agg["break"] - agg["before"],
                        "step_ci_low": np.quantile(boot["step"], 0.025),
                        "step_ci_high": np.quantile(boot["step"], 0.975),
                        "persist": agg["after"] - agg["before"],
                        "persist_ci_low": np.quantile(boot["persist"], 0.025),
                        "persist_ci_high": np.quantile(boot["persist"], 0.975),
                        **(
                            {
                                f"{k}_ci_state_{q}": np.quantile(boot_st[k], qq)
                                for k in ("excess", "step", "persist")
                                for q, qq in (("low", 0.025), ("high", 0.975))
                            }
                            if boot_st
                            else {}
                        ),
                        "excess_rel": est / ((agg["before"] + agg["after"]) / 2),
                        "step_ratio": agg["break"] / agg["before"],
                        "n_counties": int(len(common)),
                        "n_counties_wilcoxon": int(ok.sum()),
                        "county_excess_median": float(np.median(ex_c)) if ok.any() else np.nan,
                        "share_counties_positive": float((ex_c > 0).mean()) if ok.any() else np.nan,
                        "wilcoxon_p_vs_2022_23": p_before,
                        "wilcoxon_p_vs_2024_25": p_after,
                    }
                )
    return pd.DataFrame(rows)


def county_den_ok(sub: dict, common: pd.Index, den: str, min_den: int) -> np.ndarray:
    """Condados cuyo denominador de la métrica es ≥ `min_den` píxeles en los tres pares."""
    return np.all([sub[k].loc[common, den].to_numpy() >= min_den for k in sub], axis=0)


def county_threshold_table(comp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Sensibilidad de las cuotas por condado al umbral de denominador (variante NASS 30 m, CONUS)."""
    c = comp[comp.variant == "nass30"]
    sub = {
        k: c[(c.y0 == y0) & (c.y1 == y0 + 1)].set_index("zone")
        for k, y0 in (("before", 2022), ("break", 2023), ("after", 2024))
    }
    common = sub["before"].index.intersection(sub["break"].index).intersection(sub["after"].index)
    rows = []
    for metric, (num, den) in RATIO.items():
        rc = {k: sub[k].loc[common, num] / sub[k].loc[common, den] for k in sub}
        finite = np.isfinite(rc["before"]) & np.isfinite(rc["break"]) & np.isfinite(rc["after"])
        for min_den in cfg["stats"]["county_min_den_sensitivity"]:
            ok = finite.to_numpy() & county_den_ok(sub, common, den, min_den)
            ex = (rc["break"] - (rc["before"] + rc["after"]) / 2).to_numpy()[ok]
            rows.append(
                {
                    "metric": metric,
                    "min_den": min_den,
                    "n_counties": int(len(common)),
                    "n_included": int(ok.sum()),
                    "n_excluded": int(len(common) - ok.sum()),
                    "share_counties_positive": float((ex > 0).mean()) if ok.any() else np.nan,
                    "county_excess_median": float(np.median(ex)) if ok.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def placebo_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Exceso con el mismo estimador centrado en cada par de la serie (variante nass30)."""
    a = agg[(agg.STUSPS == "ALL") & (agg.variant == "nass30")].set_index("y0").sort_index()
    rows = []
    for y in a.index:
        if y - 1 in a.index and y + 1 in a.index:
            for metric in RATIO:
                rows.append(
                    {
                        "pair": f"{y}-{y + 1}",
                        "y0": y,
                        "metric": metric,
                        "excess": a.loc[y, metric] - (a.loc[y - 1, metric] + a.loc[y + 1, metric]) / 2,
                    }
                )
    return pd.DataFrame(rows)


def area_table(comp: pd.DataFrame, counties: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("nass30", "maj30", "native10", "nn30"):
        v = comp[comp.variant == variant]
        for crop in [*AREA_CODES, *(f"{c}_single" for c in AREA_CODES_SINGLE)]:
            a0 = v.assign(m2=v[f"area0_{crop}"] * v.px_m2)[["zone", "y0", "m2"]]
            a1 = v.assign(m2=v[f"area1_{crop}"] * v.px_m2)[["zone", "y1", "m2"]]
            a0, a1 = a0.rename(columns={"y0": "year"}), a1.rename(columns={"y1": "year"})
            # cada año aparece como inicial y como final de dos pares; se toma el máximo (píxeles
            # válidos en ambos años del par)
            a = pd.concat([a0, a1]).groupby(["zone", "year"], as_index=False).m2.max()
            a["acres_cdl"] = a.m2 / ACRE_M2
            a["crop"], a["variant"] = crop, variant
            rows.append(a.drop(columns="m2"))
    area = pd.concat(rows, ignore_index=True).merge(counties[["zone", "GEOID", "STUSPS"]], on="zone")
    area["qs_crop"] = area.crop.str.removesuffix("_single")  # "_single": misma referencia de Quick Stats
    q = qs.assign(acres_qs=qs.planted.where(qs.crop != "hay", qs.harvested))  # heno: cosechada
    q = q[["GEOID", "year", "crop", "acres_qs"]].rename(columns={"crop": "qs_crop"})
    return area.merge(q, on=["GEOID", "year", "qs_crop"], how="left").drop(columns="qs_crop")


def state_area(area: pd.DataFrame, qs_state: pd.DataFrame) -> pd.DataFrame:
    """Área CDL sumada por estado + Quick Stats estatal (el condal de trigo 2024 es incompleto)."""
    a = area.groupby(["STUSPS", "year", "crop", "variant"], as_index=False).acres_cdl.sum()
    fips = area[["STUSPS", "GEOID"]].drop_duplicates().assign(statefp=lambda d: d.GEOID.str[:2])
    a = a.merge(fips[["STUSPS", "statefp"]].drop_duplicates(), on="STUSPS")
    a["qs_crop"] = a.crop.str.removesuffix("_single")
    q = qs_state.assign(acres_qs=qs_state.planted.where(qs_state.crop != "hay", qs_state.harvested))
    q = q[["statefp", "year", "crop", "acres_qs"]].rename(columns={"crop": "qs_crop"})
    out = a.merge(q, on=["statefp", "year", "qs_crop"], how="left").drop(columns="qs_crop")
    return out.rename(columns={"STUSPS": "GEOID"})


def qs_did(area: pd.DataFrame, cfg: dict, min_units: int = 10) -> pd.DataFrame:
    """Exceso de Δlog(CDL/QS) en 2023→24 frente a 2022→23 y 2024→25, por cultivo y variante."""
    reps, seed = cfg["stats"]["bootstrap_reps"], cfg["seed"]
    rng = np.random.default_rng(seed)
    rows = []
    a = area.dropna(subset=["acres_qs"])
    a = a[(a.acres_qs > 0) & (a.acres_cdl > 0)]
    for (crop, variant), g in a.groupby(["crop", "variant"]):
        w = g.pivot_table(index="GEOID", columns="year", values=["acres_cdl", "acres_qs"])
        need = [2022, 2023, 2024, 2025]
        if not all((("acres_cdl", y) in w.columns) for y in need):
            continue
        w = w.dropna(subset=[(s, y) for s in ("acres_cdl", "acres_qs") for y in need])
        if len(w) < min_units:
            continue
        cdl = np.column_stack([w[("acres_cdl", y)] for y in need])
        qsv = np.column_stack([w[("acres_qs", y)] for y in need])

        def stat(ix, cdl=cdl, qsv=qsv):
            lr = np.log(cdl[ix].sum(0) / qsv[ix].sum(0))  # log-ratio agregado por año
            d = np.diff(lr)  # 22→23, 23→24, 24→25
            return d[1] - (d[0] + d[2]) / 2, lr

        est, lr = stat(np.arange(len(w)))
        boot = np.array([stat(rng.integers(0, len(w), len(w)))[0] for _ in range(reps)])
        rows.append(
            {
                "crop": crop,
                "variant": variant,
                "n_counties": len(w),
                **{f"ratio_cdl_qs_{y}": float(np.exp(lr[i])) for i, y in enumerate(need)},
                "excess_dlog_ratio": est,
                "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
            }
        )
    return pd.DataFrame(rows)


def group_flows(scope: str, counties: pd.DataFrame) -> pd.DataFrame:
    """Matriz de flujos entre grupos de cobertura (fracción de píxeles válidos) por par y variante,
    para el total y por estado; y las 15 transiciones de clase con más píxeles por par."""
    from agrors.cdl import GROUP_ORDER, group_lut

    lut = group_lut()
    suffix = "_pilot" if scope == "pilot" else ""
    st = counties.set_index("zone").STUSPS
    rows, top = [], []
    for f in sorted(TR.glob(f"*{suffix}.parquet")):
        if scope == "full" and f.stem.endswith("_pilot"):
            continue
        y0, y1, variant = f.stem.replace("_pilot", "").split("_")
        t = pd.read_parquet(f)
        t["g0"] = lut[t["from"].to_numpy()]
        t["g1"] = lut[t["to"].to_numpy()]
        t = t[(t.g0 != 255) & (t.g1 != 255)]
        t["region"] = st.reindex(t.zone).to_numpy()
        for region, g in [("ALL", t), *t.groupby("region")]:
            m = g.groupby(["g0", "g1"]).n.sum()
            tot = m.sum()
            for (a, b), v in m.items():
                rows.append(
                    {
                        "y0": int(y0),
                        "y1": int(y1),
                        "variant": variant,
                        "region": region,
                        "group_from": GROUP_ORDER[a],
                        "group_to": GROUP_ORDER[b],
                        "share": v / tot,
                    }
                )
        c = t[t["from"] != t["to"]].groupby(["from", "to"]).n.sum().sort_values(ascending=False)
        for (a, b), v in c.head(15).items():
            top.append(
                {
                    "y0": int(y0),
                    "y1": int(y1),
                    "variant": variant,
                    "from": int(a),
                    "to": int(b),
                    "share_of_valid": v / t.n.sum(),
                }
            )
    pd.DataFrame(top).to_csv(TAB / f"e1_top_transitions_{scope}.csv", index=False)
    return pd.DataFrame(rows)


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    cfg = load_config()
    counties = pd.read_csv(ROOT / "data" / "processed" / "counties.csv", dtype={"GEOID": str, "STATEFP": str})
    comp = with_old_era_variants(load_transitions(scope))
    cm = rates(comp).merge(counties[["zone", "GEOID", "STUSPS"]], on="zone")
    cm["crop_gross"] = cm.crop_turnover / cm.crop0
    keep = ["zone", "GEOID", "STUSPS", "y0", "y1", "variant", "n", *METRICS]
    cm[keep].to_csv(TAB / f"e1_county_metrics_{scope}.csv", index=False)
    agg = aggregate(comp, counties)
    agg.to_csv(TAB / f"e1_aggregate_{scope}.csv", index=False)
    ex = excess_table(comp, counties, cfg)
    ex.to_csv(TAB / f"e1_excess_{scope}.csv", index=False)
    county_threshold_table(comp, cfg).to_csv(TAB / f"e1_county_threshold_{scope}.csv", index=False)
    group_flows(scope, counties).to_csv(TAB / f"e1_group_flows_{scope}.csv", index=False)
    if scope == "full":
        placebo_table(agg).to_csv(TAB / "e1_placebo_full.csv", index=False)
    qs = pd.read_parquet(ROOT / "data" / "processed" / "quickstats_county.parquet")
    area = area_table(comp, counties, qs)
    area.to_csv(TAB / f"e1_area_county_{scope}.csv", index=False)
    sarea = state_area(area, pd.read_parquet(ROOT / "data" / "processed" / "quickstats_state.parquet"))
    sarea.to_csv(TAB / f"e1_area_state_{scope}.csv", index=False)
    did = pd.concat(
        [qs_did(area, cfg).assign(level="county"), qs_did(sarea, cfg, min_units=3).assign(level="state")]
    )
    did.to_csv(TAB / f"e1_qs_did_{scope}.csv", index=False)
    print(
        ex[ex.region == "ALL"][
            [
                "variant",
                "metric",
                "rate_2022_23",
                "rate_2023_24",
                "rate_2024_25",
                "excess",
                "ci_low",
                "ci_high",
            ]
        ].to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
