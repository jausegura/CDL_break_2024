"""Fase 4: extrae las cifras que cita el artículo a results/key_numbers.csv (id, valor, fichero, filtro).

Cada fila es trazable: `source` es un fichero de results/ y `filter` describe fila/columna.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402

TAB = ROOT / "results" / "tables"
ROWS: list[dict] = []


def add(id_: str, value, source: str, filt: str, unit: str = "") -> None:
    path = source if source.startswith("results/") else f"results/tables/{source}"
    ROWS.append({"id": id_, "value": value, "unit": unit, "source": path, "filter": filt})


def pct(x: float) -> float:
    """Porcentajes y puntos porcentuales con un decimal: la incertidumbre de clasificación y de las
    encuestas no justifica más precisión."""
    return round(100 * float(x), 1)


def ratio(x: float) -> float:
    """Cocientes, exactitudes, IoU y diferencias de log con dos decimales."""
    return round(float(x), 2)


def e1(scope: str) -> None:
    a = pd.read_csv(TAB / f"e1_aggregate_{scope}.csv")
    ex = pd.read_csv(TAB / f"e1_excess_{scope}.csv")
    for v in ("nass30", "nn30", "maj30", "native10"):
        av = a[(a.STUSPS == "ALL") & (a.variant == v)]
        for m in (
            "group_change",
            "crop_gross",
            "crop_type_change",
            "nonag_change",
            "any_change",
            "corn_corn",
            "cs_alternation",
            "crop_net",
        ):
            for y0 in (2021, 2022, 2023, 2024):
                r = av[av.y0 == y0]
                if len(r):
                    add(
                        f"{scope}.{v}.{m}.{y0}_{y0 + 1}",
                        pct(r[m].iloc[0]),
                        f"e1_aggregate_{scope}.csv",
                        f"STUSPS=ALL, variant={v}, y0={y0}, col={m}",
                        "%",
                    )
        ev = ex[(ex.region == "ALL") & (ex.variant == v)]
        for _, r in ev.iterrows():
            base = f"{scope}.{v}.{r.metric}"
            src = f"e1_excess_{scope}.csv"
            flt = f"region=ALL, variant={v}, metric={r.metric}"
            for c in (
                "excess",
                "ci_low",
                "ci_high",
                "step",
                "step_ci_low",
                "step_ci_high",
                "persist",
                "persist_ci_low",
                "persist_ci_high",
            ):
                add(f"{base}.{c}", pct(r[c]), src, f"{flt}, col={c}", "pp")
            add(
                f"{base}.share_counties_positive",
                pct(r.share_counties_positive),
                src,
                f"{flt}, col=share_counties_positive",
                "%",
            )
            add(f"{base}.n_counties", int(r.n_counties), src, f"{flt}, col=n_counties")
            add(
                f"{base}.wilcoxon_p",
                f"{r.wilcoxon_p_vs_2022_23:.1e}",
                src,
                f"{flt}, col=wilcoxon_p_vs_2022_23",
            )
    d = pd.read_csv(TAB / f"e1_qs_did_{scope}.csv")
    for _, r in d.iterrows():
        base = f"{scope}.qs.{r.level}.{r.crop}.{r.variant}"
        flt = f"level={r.level}, crop={r.crop}, variant={r.variant}"
        for c in ("excess_dlog_ratio", "ci_low", "ci_high", "ratio_cdl_qs_2023", "ratio_cdl_qs_2024"):
            add(f"{base}.{c}", ratio(r[c]), f"e1_qs_did_{scope}.csv", f"{flt}, col={c}")
        add(f"{base}.n_units", int(r.n_counties), f"e1_qs_did_{scope}.csv", f"{flt}, col=n_counties")


def regimes() -> None:
    p = TAB / "e1_regime_summary_full.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    for _, r in d[d.period == "2014-2025"].iterrows():
        for m in ("group_change", "crop_gross", "crop_type_change", "nonag_change"):
            for s in ("mean", "min", "max"):
                add(
                    f"regime.{r.regime}.{m}.{s}",
                    pct(r[f"{m}_{s}"]),
                    "e1_regime_summary_full.csv",
                    f"period=2014-2025, regime={r.regime}, col={m}_{s}",
                    "%",
                )
        add(
            f"regime.{r.regime}.n_pairs",
            int(r.n_pairs),
            "e1_regime_summary_full.csv",
            f"period=2014-2025, regime={r.regime}, col=n_pairs",
        )
    pr = pd.read_csv(TAB / "e1_pairs_regime_full.csv")
    r = pr[pr.y0 == 2018].iloc[0]
    for m in ("group_change", "crop_gross", "crop_type_change", "nonag_change"):
        add(f"pair.2018_2019.{m}", pct(r[m]), "e1_pairs_regime_full.csv", f"y0=2018, col={m}", "%")
    pl = pd.read_csv(TAB / "e1_placebo_full.csv")
    for m in ("group_change", "crop_gross", "crop_type_change", "nonag_change"):
        s = pl[(pl.metric == m) & (pl.y0 < 2022)]
        add(
            f"placebo.{m}.old_max",
            pct(s.excess.max()),
            "e1_placebo_full.csv",
            f"metric={m}, y0<2022, max(excess)",
            "pp",
        )
        add(
            f"placebo.{m}.old_max_pair",
            s.loc[s.excess.idxmax(), "pair"],
            "e1_placebo_full.csv",
            f"metric={m}, y0<2022, argmax(excess)",
        )
        add(
            f"placebo.{m}.2023_2024",
            pct(pl[(pl.metric == m) & (pl.y0 == 2023)].excess.iloc[0]),
            "e1_placebo_full.csv",
            f"metric={m}, y0=2023",
            "pp",
        )
        rk = (s.excess >= pl[(pl.metric == m) & (pl.y0 == 2023)].excess.iloc[0]).sum()
        add(
            f"placebo.{m}.n_old_pairs_ge_break",
            int(rk),
            "e1_placebo_full.csv",
            f"metric={m}, y0<2022, count(excess>=excess 2023-24)",
        )


def resampling() -> None:
    for name in ("resampling_check.csv", "verify_10m.csv"):
        p = TAB / name
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype={"statefp": str})
        if name == "resampling_check.csv":
            for _, r in d[d.statefp == "CONUS"].iterrows():
                add(
                    f"resample.{r.year}.agree_center",
                    pct(r.agree_nass_center),
                    name,
                    f"year={r.year}, statefp=CONUS, col=agree_nass_center",
                    "%",
                )
                add(
                    f"resample.{r.year}.agree_majority",
                    pct(r.agree_nass_majority),
                    name,
                    f"year={r.year}, statefp=CONUS, col=agree_nass_majority",
                    "%",
                )
                add(f"resample.{r.year}.n_pixels30", int(r.n_pixels30), name, f"year={r.year}, statefp=CONUS")
        else:
            for y, g in d.groupby("year"):
                add(
                    f"verify.{y}.official_vs_store.mean",
                    pct(g.agree_official_vs_store.mean()),
                    name,
                    f"year={y}, mean(agree_official_vs_store)",
                    "%",
                )
                add(
                    f"verify.{y}.nass30_vs_center.mean",
                    pct(g.agree_nass30_vs_center.mean()),
                    name,
                    f"year={y}, mean(agree_nass30_vs_center)",
                    "%",
                )
                add(
                    f"verify.{y}.nass30_vs_center.min",
                    pct(g.agree_nass30_vs_center.min()),
                    name,
                    f"year={y}, min(agree_nass30_vs_center)",
                    "%",
                )
                add(f"verify.{y}.n_blocks", len(g), name, f"year={y}, count")


def e2_e5() -> None:
    from agrors.io import load_config

    d = pd.read_csv(TAB / "e2_strata_pilot.csv")
    d = d[d.state.isin(load_config()["pilot"]["states"])]
    for (y0, v), g in d.groupby(["y0", "variant"]):
        for eb in (0, 1, 2):
            s = g[g.edge_bin == eb]
            add(
                f"e2.{v}.{y0}_{y0 + 1}.edge{eb}.crop_noncrop",
                pct(s.crop_noncrop_changed.sum() / s.n.sum()),
                "e2_strata_pilot.csv",
                f"variant={v}, y0={y0}, edge_bin={eb}, sum(crop_noncrop_changed)/sum(n)",
                "%",
            )
        for dec in (0, 9):
            s = g[g.size_decile == dec]
            add(
                f"e2.{v}.{y0}_{y0 + 1}.decile{dec + 1}.crop_noncrop",
                pct(s.crop_noncrop_changed.sum() / s.n.sum()),
                "e2_strata_pilot.csv",
                f"variant={v}, y0={y0}, size_decile={dec}, sum(crop_noncrop_changed)/sum(n)",
                "%",
            )
    tot = d[(d.y0 == 2023) & (d.variant == "nass30")]
    add(
        "e2.share_pixels_edge0",
        pct(tot[tot.edge_bin == 0].n.sum() / tot.n.sum()),
        "e2_strata_pilot.csv",
        "variant=nass30, y0=2023, share of n in edge_bin=0",
        "%",
    )
    add(
        "e2.share_pixels_edge2",
        pct(tot[tot.edge_bin == 2].n.sum() / tot.n.sum()),
        "e2_strata_pilot.csv",
        "variant=nass30, y0=2023, share of n in edge_bin=2",
        "%",
    )
    ex = tot.assign(x=tot.crop_noncrop_changed)
    b = d[(d.y0 == 2022) & (d.variant == "nass30")]
    exc = ex.groupby("edge_bin").x.sum() - b.groupby("edge_bin").crop_noncrop_changed.sum()
    add(
        "e2.share_excess_in_edge0",
        pct(exc[0] / exc.sum()),
        "e2_strata_pilot.csv",
        "excess crop_noncrop pixels (2023-24 minus 2022-23, nass30) in edge_bin=0 / total",
        "%",
    )
    dec = pd.read_csv(TAB / "e2_size_deciles.csv")
    add(
        "e2.decile1.acres_high",
        round(float(dec.acres_high.iloc[0]), 1),
        "e2_size_deciles.csv",
        "decile=0, acres_high",
        "acres",
    )
    add(
        "e2.decile10.acres_low",
        round(float(dec.acres_low.iloc[9]), 1),
        "e2_size_deciles.csv",
        "decile=9, acres_low",
        "acres",
    )
    add("e2.n_polygons", int(dec.n_polygons.sum()), "e2_size_deciles.csv", "sum(n_polygons)")
    e5 = pd.read_csv(TAB / "e5_harmonization.csv")
    for _, r in e5[e5.region == "ALL"].iterrows():
        base = f"e5.{r.rule}.{r.metric}"
        add(
            f"{base}.step",
            pct(r.step),
            "e5_harmonization.csv",
            f"region=ALL, rule={r.rule}, metric={r.metric}, col=step",
            "pp",
        )
        add(
            f"{base}.step_reduction",
            pct(r.step_reduction_vs_R0),
            "e5_harmonization.csv",
            f"region=ALL, rule={r.rule}, metric={r.metric}, col=step_reduction_vs_R0",
            "%",
        )
        add(
            f"{base}.persist_reduction",
            pct(r.persist_reduction_vs_R0),
            "e5_harmonization.csv",
            f"region=ALL, rule={r.rule}, metric={r.metric}, col=persist_reduction_vs_R0",
            "%",
        )


def e3() -> None:
    m = pd.read_csv(TAB / "e3_csb_match.csv")
    for c in ("label_agree_2023", "iou_median", "share_new_unmatched", "share_new_unmatched_area"):
        for st, v in zip(m.state, m[c], strict=True):
            val, unit = (ratio(v), "") if c == "iou_median" else (pct(v), "%")
            add(f"e3.{st}.{c}", val, "e3_csb_match.csv", f"state={st}, col={c}", unit)
    lc = pd.read_csv(TAB / "e3_csb_label_change.csv")
    for w, g in lc.groupby("window"):
        for (y0, y1), s in g.groupby(["y0", "y1"]):
            if y0 >= 2021:
                wa = np.average(s.change_area_share, weights=s.corn_acres_y0 * 0 + 1)
                add(
                    f"e3.{w}.{y0}_{y1}.change_area_share_meanstates",
                    pct(wa),
                    "e3_csb_label_change.csv",
                    f"window={w}, y0={y0}, unweighted mean over states of change_area_share",
                    "%",
                )
    v = pd.read_csv(TAB / "e3_csb_versions.csv")
    for w, g in v.groupby("window"):
        add(
            f"e3.{w}.n_polygons_pilot",
            int(g.n_polygons.sum()),
            "e3_csb_versions.csv",
            f"window={w}, sum(n_polygons)",
        )


def e4() -> None:
    p = TAB / "e4_metrics.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    g = d.groupby(["train", "test"])
    for (tr, te), s in g:
        for c in ("oa", "f1_macro"):
            add(
                f"e4.{tr}.{te}.{c}",
                ratio(s[c].mean()),
                "e4_metrics.csv",
                f"train={tr}, test={te}, mean over seeds of {c}",
            )
        add(f"e4.{tr}.{te}.n_test", int(s.n_test.iloc[0]), "e4_metrics.csv", f"train={tr}, test={te}, n_test")


def accuracy() -> None:
    d = pd.read_csv(TAB / "official_accuracy.csv")
    for (sc, y), g in d.groupby(["scope", "year"]):
        add(
            f"acc.{sc}.{y}.oa",
            float(g.overall_accuracy.iloc[0]),
            "official_accuracy.csv",
            f"scope={sc}, year={y}, overall_accuracy",
            "%",
        )


def extras() -> None:
    """Cifras derivadas (cocientes, inventario, sensibilidad, transiciones dominantes)."""
    a = pd.read_csv(TAB / "e1_aggregate_full.csv")
    a = a[(a.STUSPS == "ALL") & (a.variant == "nass30")].set_index("y0")
    for m in ("group_change", "crop_gross", "nonag_change"):
        add(
            f"derived.{m}.fold_2324_vs_2223",
            round(float(a.loc[2023, m] / a.loc[2022, m]), 1),
            "e1_aggregate_full.csv",
            f"STUSPS=ALL, variant=nass30, {m} y0=2023 / y0=2022",
            "x",
        )
    add(
        "derived.full.area_mkm2",
        round(float(a.loc[2022, "n"]) * 900 / 1e12, 2),
        "e1_aggregate_full.csv",
        "STUSPS=ALL, variant=nass30, y0=2022, n*900 m2 / 1e12",
        "million km2",
    )
    inv = pd.read_csv(TAB / "data_inventory.csv")
    for _, r in inv.iterrows():
        add(
            f"inv.{r.dataset}.{r['item']}".replace(" ", "_"),
            r.value,
            "data_inventory.csv",
            f"dataset={r.dataset}, item={r['item']}",
        )
    cd = pd.read_csv(TAB / "e1_cropdef_sensitivity_full.csv")
    for _, r in cd[cd.y0 >= 2017].iterrows():
        for c in ("crop_gross", "crop_net"):
            add(
                f"cropdef.{r.definition}.{c}.{r.y0}_{r.y1}",
                pct(r[c]),
                "e1_cropdef_sensitivity_full.csv",
                f"definition={r.definition}, y0={r.y0}, col={c}",
                "%",
            )
    top = pd.read_csv(TAB / "e1_top_transitions_full.csv")
    top = top[top.variant == "nass30"]
    for y0 in (2022, 2023, 2024):
        for f_, t_ in ((37, 176), (152, 176), (1, 5), (5, 1)):
            r = top[(top.y0 == y0) & (top["from"] == f_) & (top.to == t_)]
            if len(r):
                add(
                    f"top.{f_}_to_{t_}.{y0}_{y0 + 1}",
                    pct(r.share_of_valid.iloc[0]),
                    "e1_top_transitions_full.csv",
                    f"variant=nass30, y0={y0}, from={f_}, to={t_}, col=share_of_valid",
                    "%",
                )
    gf = pd.read_csv(TAB / "e1_group_flows_full.csv")
    gf = gf[(gf.region == "ALL") & (gf.variant == "nass30")]
    for y0 in (2022, 2023, 2024):
        for g0, g1 in (
            ("crop", "grass_pasture"),
            ("grass_pasture", "crop"),
            ("forest_shrub", "grass_pasture"),
        ):
            r = gf[(gf.y0 == y0) & (gf.group_from == g0) & (gf.group_to == g1)]
            add(
                f"flow.{g0}_to_{g1}.{y0}_{y0 + 1}",
                pct(r.share.iloc[0]),
                "e1_group_flows_full.csv",
                f"region=ALL, variant=nass30, y0={y0}, {g0}->{g1}, col=share",
                "%",
            )
    cnt = pd.read_csv(TAB / "cropsight_counts_state_year.csv", index_col=0)
    for st in ("IA", "IL", "PA", "CA"):
        add(
            f"cropsight.{st}.2023",
            int(cnt.loc[st, "2023"]),
            "cropsight_counts_state_year.csv",
            f"STUSPS={st}, 2023",
        )


def complementary() -> None:
    """Cifras complementarias (áreas en acres, contraste con Quick Stats y robustez)."""
    from agrors.quickstats import ACRE_M2

    ex = pd.read_csv(TAB / "e1_excess_full.csv")
    ex = ex[ex.region == "ALL"]
    for _, r in ex.iterrows():
        base = f"full.{r.variant}.{r.metric}"
        add(
            f"{base}.n_counties_wilcoxon",
            int(r.n_counties_wilcoxon),
            "e1_excess_full.csv",
            f"region=ALL, variant={r.variant}, metric={r.metric}, col=n_counties_wilcoxon",
        )
        for k in ("excess", "step", "persist"):
            for q in ("low", "high"):
                c = f"{k}_ci_state_{q}"
                if c in r and pd.notna(r[c]):
                    add(
                        f"{base}.{c}",
                        pct(r[c]),
                        "e1_excess_full.csv",
                        f"region=ALL, variant={r.variant}, metric={r.metric}, col={c}",
                        "pp",
                    )
    a = pd.read_csv(TAB / "e1_aggregate_full.csv")
    a = a[(a.STUSPS == "ALL") & (a.variant == "nass30")].set_index("y0")
    for y0 in (2022, 2023, 2024):
        dm = (a.loc[y0, "crop1"] - a.loc[y0, "crop0"]) * 900 / ACRE_M2 / 1e6
        add(
            f"derived.crop_net_macres.{y0}_{y0 + 1}",
            round(float(dm), 1),
            "e1_aggregate_full.csv",
            f"STUSPS=ALL, variant=nass30, y0={y0}, (crop1-crop0)*900 m2 / acre / 1e6",
            "M acres",
        )
        add(
            f"derived.crop_net_macres_abs.{y0}_{y0 + 1}",
            round(abs(float(dm)), 1),
            "e1_aggregate_full.csv",
            f"STUSPS=ALL, variant=nass30, y0={y0}, |crop1-crop0|*900 m2 / acre / 1e6",
            "M acres",
        )
    pr = pd.read_csv(TAB / "e1_pairs_regime_full.csv")
    brk = pr[pr.y0 == 2023].iloc[0]
    for m in ("crop_gross", "group_change"):
        ge = pr[(pr.y0 != 2023) & (pr[m] >= brk[m])]
        add(
            f"rank.{m}.n_pairs_ge_break",
            len(ge),
            "e1_pairs_regime_full.csv",
            f"count(y0!=2023 & {m} >= 2023-24)",
        )
        add(
            f"rank.{m}.pairs_ge_break",
            ", ".join(f"{a}→{b}" for a, b in zip(ge.y0, ge.y1, strict=True)) or "none",
            "e1_pairs_regime_full.csv",
            f"pairs with {m} >= 2023-24",
        )
    dec = pd.read_csv(TAB / "e2_size_deciles.csv")
    add(
        "e2.decile1.acres_low",
        round(float(dec.acres_low.iloc[0]), 1),
        "e2_size_deciles.csv",
        "decile=0, acres_low",
        "acres",
    )
    early = pr[pr.y0 <= 2012]
    for m in ("crop_gross", "group_change"):
        add(f"early.{m}.max", pct(early[m].max()), "e1_pairs_regime_full.csv", f"y0<=2012, max({m})", "%")
        add(f"early.{m}.min", pct(early[m].min()), "e1_pairs_regime_full.csv", f"y0<=2012, min({m})", "%")
    e5 = pd.read_csv(TAB / "e5_harmonization.csv")
    e5 = e5[e5.metric == "crop_noncrop"]
    for _, r in e5[e5.rule.isin(["R1_majority", "R3_edge60"])].iterrows():
        add(
            f"e5.{r.region}.{r.rule}.step_reduction",
            pct(r.step_reduction_vs_R0),
            "e5_harmonization.csv",
            f"region={r.region}, rule={r.rule}, metric=crop_noncrop, col=step_reduction_vs_R0",
            "%",
        )
    e2 = pd.read_csv(TAB / "e2_strata_pilot.csv")
    e2 = e2[e2.state.isin(["IA", "IL", "NE", "KS", "PA", "CA"])]
    for y0 in (2022, 2023):
        for eb in (0, 2):
            s_ = e2[(e2.y0 == y0) & (e2.variant == "nass30") & (e2.edge_bin == eb)]
            add(
                f"e2.ratio_base.edge{eb}.{y0}",
                pct(s_.crop_noncrop_changed.sum() / s_.n.sum()),
                "e2_strata_pilot.csv",
                f"variant=nass30, y0={y0}, edge_bin={eb}",
                "%",
            )
    lc = pd.read_csv(TAB / "e3_csb_label_change.csv")
    lc = lc[lc.window == "2018-2025"]
    for st in ("IA", "IL", "NE", "KS", "PA", "CA"):
        for y0 in (2022, 2023):
            r = lc[(lc.state == st) & (lc.y0 == y0)].iloc[0]
            add(
                f"e3.{st}.change_area_share.{y0}_{y0 + 1}",
                pct(r.change_area_share),
                "e3_csb_label_change.csv",
                f"window=2018-2025, state={st}, y0={y0}, col=change_area_share",
                "%",
            )
        for y0 in (2023, 2024):
            r = lc[(lc.state == st) & (lc.y0 == y0)].iloc[0]
            add(
                f"e3.{st}.corn_macres.{y0}",
                round(float(r.corn_acres_y0) / 1e6, 1),
                "e3_csb_label_change.csv",
                f"window=2018-2025, state={st}, y0={y0}, col=corn_acres_y0/1e6",
                "M acres",
            )


def sensitivity() -> None:
    """Cifras de los análisis de sensibilidad (definición de clases, umbrales, contrafactuales, CSB)."""
    f = pd.read_csv(TAB / "crosswalk_frequency.csv")
    for c, g in f.groupby("code"):
        add(
            f"xw.freq.{c}.max",
            round(float(g.share_pct.max()), 2),
            "crosswalk_frequency.csv",
            f"code={c}, max(share_pct)",
            "%",
        )
        add(
            f"xw.freq.{c}.pixels_total",
            int(g.pixels.sum()),
            "crosswalk_frequency.csv",
            f"code={c}, sum(pixels)",
        )
    sens = pd.read_csv(TAB / "crosswalk_sensitivity.csv")
    for _, r in sens[sens.variant == "nass30"].iterrows():
        flt = f"variant=nass30, crosswalk={r.crosswalk}, metric={r.metric}"
        for c in ("excess", "step", "rate_2023_2024"):
            add(
                f"xw.{r.crosswalk}.{r.metric}.{c}",
                round(float(r[c]), 1),
                "crosswalk_sensitivity.csv",
                f"{flt}, col={c}",
                "pp",
            )
        if r.crosswalk == "usda":
            add(
                f"xw.max_diff.{r.metric}",
                round(float(r.max_abs_diff_any_pair), 2),
                "crosswalk_sensitivity.csv",
                f"{flt}, col=max_abs_diff_any_pair",
                "pp",
            )
    add(
        "xw.max_diff.all",
        round(float(sens[sens.variant == "nass30"].max_abs_diff_any_pair.max()), 2),
        "crosswalk_sensitivity.csv",
        "variant=nass30, max(max_abs_diff_any_pair)",
        "pp",
    )
    cf = pd.read_csv(TAB / "e1_counterfactuals_full.csv")
    for _, r in cf.iterrows():
        for c in ("estimate_pp", "ci_low", "ci_high"):
            add(
                f"cf.{r.metric}.{r.estimand}.{c}",
                round(float(r[c]), 1),
                "e1_counterfactuals_full.csv",
                f"metric={r.metric}, estimand={r.estimand}, col={c}",
                "pp",
            )
    for m, g in cf.groupby("metric"):
        b = g[g.estimand.str.startswith("break_minus")]
        add(
            f"cf.{m}.min",
            round(float(b.estimate_pp.min()), 1),
            "e1_counterfactuals_full.csv",
            f"metric={m}, min over break_minus_* estimands",
            "pp",
        )
        add(
            f"cf.{m}.max",
            round(float(b.estimate_pp.max()), 1),
            "e1_counterfactuals_full.csv",
            f"metric={m}, max over break_minus_* estimands",
            "pp",
        )
    add(
        "cf.pairs_nlcd_switch",
        cf.pairs_nlcd_switch.iloc[0].replace(" ", ", "),
        "e1_counterfactuals_full.csv",
        "pairs_nlcd_switch",
    )
    th = pd.read_csv(TAB / "e1_county_threshold_full.csv")
    for _, r in th.iterrows():
        base = f"thr.{r.metric}.{r.min_den}"
        add(
            f"{base}.n_excluded",
            int(r.n_excluded),
            "e1_county_threshold_full.csv",
            f"metric={r.metric}, min_den={r.min_den}",
        )
        add(
            f"{base}.share_positive",
            pct(r.share_counties_positive),
            "e1_county_threshold_full.csv",
            f"metric={r.metric}, min_den={r.min_den}, col=share_counties_positive",
            "%",
        )
    add("thr.n_counties", int(th.n_counties.iloc[0]), "e1_county_threshold_full.csv", "n_counties")
    pv = pd.read_csv(TAB / "provenance_offsets.csv")
    for y, g in pv.groupby("year"):
        off = g[[c for c in g.columns if c.startswith("agree_r")]].mean()
        add(
            f"prov.{y}.best_offset",
            pct(off.max()),
            "provenance_offsets.csv",
            f"year={y}, max over offsets of mean(agree_rXcY)",
            "%",
        )
        add(
            f"prov.{y}.best_offset_name",
            off.idxmax().removeprefix("agree_"),
            "provenance_offsets.csv",
            f"year={y}, argmax offset",
        )
        add(
            f"prov.{y}.center",
            pct(off["agree_r1c1"]),
            "provenance_offsets.csv",
            f"year={y}, mean(agree_r1c1)",
            "%",
        )
        for c in ("agree_majority", "agree_homogeneous", "share_homogeneous", "nodata_mismatch"):
            add(f"prov.{y}.{c}", pct(g[c].mean()), "provenance_offsets.csv", f"year={y}, mean({c})", "%")
        add(f"prov.{y}.n_blocks", len(g), "provenance_offsets.csv", f"year={y}, count")
    ab = pd.read_csv(TAB / "accuracy_bound.csv")
    for _, r in ab.iterrows():
        st = "CONUS" if r.state.startswith("CONUS") else r.state
        for c in ("bound_pct", "observed_pct"):
            add(
                f"accb.{st}.{r.pair}.{c.removesuffix('_pct')}",
                pct(r[c] / 100),
                "accuracy_bound.csv",
                f"state={r.state}, pair={r.pair}, col={c}",
                "%",
            )
        if st == "CONUS":
            add(
                f"accb.CONUS.{r.pair}.ratio",
                ratio(r.observed_pct / r.bound_pct),
                "accuracy_bound.csv",
                f"state={r.state}, pair={r.pair}, observed_pct/bound_pct",
            )
    add("accb.n_states", int(ab.n_states.dropna().iloc[0]), "accuracy_bound.csv", "n_states (CONUS rows)")
    fa = pd.read_csv(TAB / "accuracy_forage_classes.csv")
    for _, r in fa.iterrows():
        add(
            f"acc.{r.scope}.{r.year}.{r.code}.pa",
            round(float(r.producer_acc), 1),
            "accuracy_forage_classes.csv",
            f"scope={r.scope}, year={r.year}, code={r.code}, col=producer_acc",
            "%",
        )
    rs = pd.read_csv(ROOT / "results" / "reference_sample" / "strata_2023_2024.csv")
    src = "results/reference_sample/strata_2023_2024.csv"
    add("refs.n_points", int(rs.n.sum()), src, "sum(n)")
    add("refs.n_strata", len(rs), src, "count(stratum)")
    add("refs.n_tiles", int(rs.n_tiles.iloc[0]), src, "n_tiles")
    add("refs.n_min", int(rs.n.min()), src, "min(n)")
    add("refs.n_max", int(rs.n.max()), src, "max(n)")
    e4 = pd.read_csv(TAB / "e4_metrics.csv")
    if "n_objects" in e4:
        for (tr, te), g in e4.dropna(subset=["n_objects"]).groupby(["train", "test"]):
            src, flt = "e4_metrics.csv", f"train={tr}, test={te}"
            add(f"e4.{tr}.{te}.n_objects", int(g.n_objects.iloc[0]), src, f"{flt}, n_objects")
            for c in ("oa_object", "oa_ci_low_obj", "oa_ci_high_obj"):
                add(f"e4.{tr}.{te}.{c}", ratio(g[c].mean()), src, f"{flt}, mean over seeds of {c}")


def robustness() -> None:
    """Cifras de borde × tamaño, matrices de error, conversión en acres y CV de Quick Stats."""
    e5 = pd.read_csv(TAB / "e5_harmonization.csv")
    e5 = e5[e5.metric == "crop_noncrop"]
    for _, r in e5.iterrows():
        flt = f"region={r.region}, rule={r.rule}, metric=crop_noncrop"
        add(
            f"e5x.{r.region}.{r.rule}.excess",
            pct(r.excess),
            "e5_harmonization.csv",
            f"{flt}, col=excess",
            "pp",
        )
        add(
            f"e5x.{r.region}.{r.rule}.excess_reduction",
            pct(r.excess_reduction_vs_R0),
            "e5_harmonization.csv",
            f"{flt}, col=excess_reduction_vs_R0",
            "%",
        )
    m = pd.read_csv(TAB / "e3_csb_match.csv")
    for _, r in m.iterrows():
        for c in ("share_old_unmatched", "share_old_with_multiple_new", "share_new_unmatched"):
            add(f"e3.{r.state}.{c}", pct(r[c]), "e3_csb_match.csv", f"state={r.state}, col={c}", "%")
        for c in ("iou_median_ci_low", "iou_median_ci_high"):
            add(f"e3.{r.state}.{c}", ratio(r[c]), "e3_csb_match.csv", f"state={r.state}, col={c}")
    add("e3.iou_n", int(m.iou_n.iloc[0]), "e3_csb_match.csv", "iou_n")
    ce = pd.read_csv(TAB / "e1_cropdef_estimators_full.csv")
    for _, r in ce.iterrows():
        for c in (
            "rate_2022_2023",
            "rate_2023_2024",
            "rate_2024_2025",
            "excess",
            "excess_ci_low",
            "excess_ci_high",
            "step",
            "persist",
            "share_counties_positive",
        ):
            add(
                f"cde.{r.definition}.{r.metric}.{c}",
                round(float(r[c]), 1),
                "e1_cropdef_estimators_full.csv",
                f"definition={r.definition}, metric={r.metric}, col={c}",
                "%" if c.startswith(("rate", "share")) else "pp",
            )
    cs = pd.read_csv(TAB / "e1_cropdef_sensitivity_full.csv")
    for _, r in cs[cs.y0 >= 2022].iterrows():
        add(
            f"cropdef.{r.definition}.net_macres.{r.y0}_{r.y1}",
            round(float(r.crop_net_macres), 1),
            "e1_cropdef_sensitivity_full.csv",
            f"definition={r.definition}, y0={r.y0}, col=crop_net_macres",
            "M acres",
        )
    cv = pd.read_csv(TAB / "conversion_acres_full.csv").set_index("y0")
    for y0 in (2022, 2023, 2024):
        for k in ("grass_to_crop", "crop_to_grass"):
            for c in ("macres", "excess_macres"):
                add(
                    f"conv.{k}.{y0}_{y0 + 1}.{c}",
                    round(float(cv.loc[y0, f"{k}_{c}"]), 1),
                    "conversion_acres_full.csv",
                    f"y0={y0}, col={k}_{c}",
                    "M acres",
                )
    for k in ("grass_to_crop", "crop_to_grass"):
        add(
            f"conv.{k}.old_mean",
            round(float(cv[f"{k}_old_mean_macres"].iloc[0]), 1),
            "conversion_acres_full.csv",
            f"col={k}_old_mean_macres (mean of pairs 2014-2015 to 2022-2023)",
            "M acres",
        )
    lit = pd.read_csv(ROOT / "paper" / "literature_numbers.csv").set_index("id")
    wwf = float(lit.loc["wwf2026.conversion_2023_macres", "value"])
    add(
        "conv.ratio_reconversion_to_wwf",
        round(float(cv.loc[2024, "grass_to_crop_excess_macres"]) / wwf, 1),
        "conversion_acres_full.csv",
        "y0=2024 grass_to_crop_excess_macres / paper/literature_numbers.csv wwf2026.conversion_2023_macres",
        "x",
    )
    em = pd.read_csv(TAB / "e2_edge_size_model.csv")
    for _, r in em.iterrows():
        t = r.term.replace("<", "lt").replace(">=", "ge").replace(" m", "m").replace(" ", "")
        for c in ("coef_pp", "loo_min", "loo_max"):
            add(
                f"es.{r.model}.{t}.{c}",
                round(float(r[c]), 1),
                "e2_edge_size_model.csv",
                f"model={r.model}, term={r.term}, col={c}",
                "pp",
            )
    ec = pd.read_csv(TAB / "e2_edge_size_cross.csv")
    for (eb, dec), r in ec.set_index(["edge_bin", "size_decile"]).iterrows():
        if dec in (1, 2, 10):
            add(
                f"es.cross.edge{eb}.decile{dec}.step",
                round(float(r.step), 1),
                "e2_edge_size_cross.csv",
                f"edge_bin={eb}, size_decile={dec}, col=step",
                "pp",
            )
    qc = pd.read_csv(TAB / "qs_cv_county.csv")
    add("qscv.median_min", round(float(qc.cv_median.min()), 1), "qs_cv_county.csv", "min(cv_median)", "%")
    add("qscv.median_max", round(float(qc.cv_median.max()), 1), "qs_cv_county.csv", "max(cv_median)", "%")
    add(
        "qscv.share_with_cv_min",
        pct(qc.share_with_cv.min() / 100),
        "qs_cv_county.csv",
        "min(share_with_cv)",
        "%",
    )
    fs = pd.read_csv(TAB / "fig4_support_full.csv")
    for _, r in fs.iterrows():
        add(
            f"f4.{r.crop}.n_states",
            int(r.n_states_fixed),
            "fig4_support_full.csv",
            f"crop={r.crop}, n_states_fixed",
        )
        add(
            f"f4.{r.crop}.share_min",
            pct(r.share_survey_area_min),
            "fig4_support_full.csv",
            f"crop={r.crop}, share_survey_area_min",
            "%",
        )


def main() -> int:
    for scope in ("pilot", "full"):
        e1(scope)
    regimes()
    resampling()
    e2_e5()
    e3()
    e4()
    accuracy()
    extras()
    complementary()
    sensitivity()
    robustness()
    out = pd.DataFrame(ROWS)
    out.to_csv(ROOT / "results" / "key_numbers.csv", index=False)
    print(len(out), "cifras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
