"""Fase 4: figuras del artículo (PDF vectorial + PNG 600 dpi) en results/figures/.

fig1_county_excess      mapa por condado del exceso 2023→24 (cambio de grupo de cobertura, nass30)
fig2_change_series      serie nacional 2008–2025 de cuatro métricas, con la era nueva sombreada
fig3_edge_size          cambio cultivo↔no cultivo por banda de borde y decil de tamaño CSB (piloto)
fig4_rotation_area      rotaciones (a), razón CDL/Quick Stats de cultivos principales (b) y heno (c)
Uso: python scripts/04a_figures.py [pilot|full]  (escala de E1; E2/E3 siempre piloto)
Estilo: src/agrors/plotting.py (anchos Elsevier, Source Sans 3, paleta validada).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT, load_config  # noqa: E402
from agrors.plotting import (  # noqa: E402
    BLUE,
    DIVERGING,
    GREY,
    INK,
    INK2,
    MUTED,
    TEAL,
    TERRACOTTA,
    VARIANT_LABEL,
    VARIANT_STYLE,
    WIDTH_DOUBLE,
    end_label,
    end_labels,
    panel_label,
    save,
    setup,
    shade_new_era,
)

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


def _year_axis(ax, bottom: bool = True) -> None:
    ax.set_xlim(2008.4, 2025.9)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    if not bottom:
        ax.tick_params(labelbottom=False)


def fig1(scope: str) -> None:
    cm = pd.read_csv(TAB / f"e1_county_metrics_{scope}.csv", dtype={"GEOID": str})
    cm = cm[cm.variant == "nass30"]
    w = cm.pivot_table(index="GEOID", columns="y0", values="group_change")
    w = w[[2022, 2023, 2024]].dropna()
    ex = (100 * (w[2023] - (w[2022] + w[2024]) / 2)).rename("excess")  # puntos porcentuales
    g = gpd.read_file(ROOT / "data" / "interim" / "zones" / "counties_5070.gpkg")
    g = g.merge(ex, left_on="GEOID", right_index=True, how="inner")
    states = g.dissolve("STUSPS")
    pilot = states.loc[states.index.isin(load_config()["pilot"]["states"])]
    nation = states.dissolve()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.6))
    norm = TwoSlopeNorm(vcenter=0, vmin=-5, vmax=15)
    g.plot(column="excess", ax=ax, cmap=DIVERGING, norm=norm, linewidth=0, rasterized=True)
    states.boundary.plot(ax=ax, color="white", linewidth=0.35)
    pilot.boundary.plot(ax=ax, color=INK, linewidth=0.75)
    nation.boundary.plot(ax=ax, color=INK2, linewidth=0.45)
    ax.set_axis_off()
    ax.set_aspect("equal")
    # barra de color fina en la esquina inferior izquierda
    cax = ax.inset_axes([0.05, 0.07, 0.30, 0.025])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=[-5, 0, 5, 10, 15])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.ax.set_title(
        "Excess land-cover group change, 2023–2024 (pp)", fontsize=7, color=INK, loc="left", pad=4
    )
    ax.legend(
        handles=[Line2D([0], [0], color=INK, lw=0.75, label="Pilot states")],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        fontsize=6.8,
        handlelength=1.6,
    )
    save(fig, f"fig1_county_excess_{scope}", FIG)


def fig2(scope: str) -> None:
    nlcd = {int(k): str(v) for k, v in load_config()["nlcd_training"].items()}
    a = pd.read_csv(TAB / f"e1_aggregate_{scope}.csv")
    a = a[a.STUSPS == "ALL"]
    panels = [
        ("group_change", "Land-cover group change", "% of pixels", True),
        ("crop_gross", "Gross cropland turnover", "% of cropland", True),
        ("crop_type_change", "Crop-type change in stable cropland", "% of stable cropland", False),
        ("nonag_change", "Change in forest, developed and water", "% of class pixels", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.62), sharex=True)
    base = a[a.variant == "nass30"].sort_values("y0")
    switch = np.array([nlcd[y0] != nlcd[y1] for y0, y1 in zip(base.y0, base.y1, strict=True)])
    for k, (ax, (m, title, ylab, zero)) in enumerate(zip(axes.ravel(), panels, strict=True)):
        shade_new_era(ax, label=(k == 0))
        y = base[m].to_numpy() * 100
        x = base.y1.to_numpy()
        ax.plot(x, y, color=BLUE, lw=1.4, zorder=3)
        ax.plot(x[~switch], y[~switch], "o", color=BLUE, ms=3.6, zorder=4)
        ax.plot(x[switch], y[switch], "o", mfc="white", mec=BLUE, mew=1.1, ms=4.2, zorder=4)
        for v, dx in (("maj30", 0.2), ("native10", 0.4), ("nn30", 0.6)):
            s_ = a[(a.variant == v) & (a.y1 >= 2024)].sort_values("y0")
            ax.plot(
                s_.y1 + dx,
                s_[m] * 100,
                ms=4.2,
                ls="none",
                zorder=5,
                **VARIANT_STYLE[v],
            )
        panel_label(ax, "abcd"[k], title)
        ax.set_ylabel(ylab)
        if zero:
            ax.set_ylim(bottom=0)
        _year_axis(ax, bottom=(k >= 2))
    for ax in axes[1]:
        ax.set_xlabel("Second year of the year pair")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", ms=3.6, lw=1.4, label=VARIANT_LABEL["nass30"]),
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            mfc="white",
            mew=1.1,
            ms=4.2,
            lw=0,
            label="NLCD training edition changed",
        ),
        *(
            Line2D([0], [0], ms=4.2, lw=0, label=VARIANT_LABEL[v], **VARIANT_STYLE[v])
            for v in ("maj30", "native10", "nn30")
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=6.8,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6, w_pad=2.0)
    save(fig, f"fig2_change_series_{scope}", FIG)


def fig3() -> None:
    p = TAB / "e2_strata_pilot.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.state.isin(load_config()["pilot"]["states"]) & (d.variant == "nass30")]
    series = [
        ((2022, 2023), GREY, "2022–2023"),
        ((2023, 2024), TERRACOTTA, "2023–2024"),
        ((2024, 2025), BLUE, "2024–2025"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.36), sharey=True, gridspec_kw={"width_ratios": [1, 2.2]}
    )
    labels3 = []
    for (y0, y1), color, lab in series:
        s = d[(d.y0 == y0) & (d.y1 == y1)]
        e = s.groupby("edge_bin")[["n", "crop_noncrop_changed"]].sum()
        ye = (e.crop_noncrop_changed / e.n * 100).to_numpy()
        axes[0].plot(range(3), ye, marker="o", color=color, ms=3.6)
        z = s.groupby("size_decile")[["n", "crop_noncrop_changed"]].sum()
        yz = (z.crop_noncrop_changed / z.n * 100).to_numpy()
        axes[1].plot(z.index + 1, yz, marker="o", color=color, ms=3.6)
        labels3.append((yz[-1], lab, color))
    end_labels(axes[1], 10, labels3, min_gap=2.2, dx=0.42)
    axes[0].set_xticks(range(3), ["<30 m", "30–60 m", "≥60 m"])
    axes[0].set_xlim(-0.3, 2.3)
    axes[0].set_xlabel("Distance to field edge")
    axes[0].set_ylabel("Crop ↔ non-crop switching\n(% of field pixels)")
    axes[1].set_xticks(range(1, 11))
    axes[1].set_xlim(0.6, 10.5)
    axes[1].set_xlabel("Field-size decile (1 = smallest)")
    axes[0].set_ylim(bottom=0)
    panel_label(axes[0], "a", "By distance to the CSB field edge", x=-0.06)
    panel_label(axes[1], "b", "By CSB field-size decile", x=-0.03)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig3_edge_size", FIG)


def fig4(scope: str) -> None:
    a = pd.read_csv(TAB / f"e1_aggregate_{scope}.csv")
    a = a[(a.STUSPS == "ALL") & (a.variant == "nass30")].sort_values("y0")
    fig, axes = plt.subplots(
        1, 3, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.33), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]}
    )
    # (a) rotaciones
    ax = axes[0]
    shade_new_era(ax)
    ax.plot(a.y1, a.cs_alternation * 100, marker="o", color=TERRACOTTA, ms=3.2)
    ax.plot(a.y1, a.corn_corn * 100, marker="o", color=BLUE, ms=3.2)
    csb = TAB / "e3_csb_label_change.csv"
    if csb.exists():
        c = pd.read_csv(csb)
        c = c[c.window == "2018-2025"]
        c = c.groupby("y1").apply(
            lambda g: np.average(g.corn_corn, weights=g.corn_acres_y0), include_groups=False
        )
        ax.plot(
            c.index,
            c.to_numpy() * 100,
            marker="s",
            ms=2.8,
            ls=(0, (2, 1.5)),
            lw=1.0,
            color=BLUE,
            mfc="white",
            mew=0.8,
        )
        ax.text(2015.6, 44, "corn → corn,\nCSB polygons", fontsize=6.8, color=BLUE, va="center", ha="left")
    end_label(ax, 2008.6, a.cs_alternation.iloc[0] * 100, "corn ↔ soybean", TERRACOTTA, dy=11, dx=0)
    end_label(ax, 2008.6, a.corn_corn.iloc[0] * 100, "corn → corn", BLUE, dy=-6.5, dx=0)
    ax.set_ylim(0, 72)
    ax.set_ylabel("% of first-year pixels")
    ax.set_xlabel("Second year of the year pair")
    _year_axis(ax)
    panel_label(ax, "a", "Rotation metrics")
    # (b) cultivos principales frente a Quick Stats; (c) heno
    allst = pd.read_csv(TAB / f"e1_area_state_{scope}.csv")
    allst = allst[allst.variant == "nass30"]
    did = allst.dropna(subset=["acres_qs"])
    years = sorted(did.year.unique())
    support = []

    def ratio(crop):
        """Cociente CDL/encuesta sumado sobre un conjunto fijo de estados: los que tienen estimación de
        la encuesta en todos los años de la serie (se registra su cobertura en fig4_support)."""
        c = did[did.crop == crop]
        fixed = c.groupby("GEOID").year.nunique()
        fixed = fixed[fixed == len(years)].index
        s = c[c.GEOID.isin(fixed)].groupby("year")[["acres_cdl", "acres_qs"]].sum()
        tot = c.groupby("year").acres_qs.sum()
        support.append(
            {
                "crop": crop,
                "n_states_fixed": len(fixed),
                "share_survey_area_min": float((s.acres_qs / tot).min()),
                "share_survey_area_mean": float((s.acres_qs / tot).mean()),
            }
        )
        return s.index.to_numpy(), (s.acres_cdl / s.acres_qs).to_numpy()

    ax = axes[1]
    shade_new_era(ax)
    ax.axhline(1, color=MUTED, lw=0.6, zorder=1)
    labels4 = []
    for crop, color, lab in (
        ("corn", BLUE, "corn"),
        ("soybeans", TERRACOTTA, "soybeans"),
        ("winter_wheat", TEAL, "winter wheat"),
    ):
        x, y = ratio(crop)
        ax.plot(x, y, marker="o", color=color, ms=3.0)
        labels4.append((y[-1], lab, color))
    end_labels(ax, x[-1], labels4, min_gap=0.014, dx=0.35)
    ax.set_ylabel("CDL area / survey area")
    ax.set_xlabel("Year")
    _year_axis(ax)
    ax.set_xlim(2007.5, 2028.6)
    panel_label(ax, "b", "Major crops (planted area)")
    ax = axes[2]
    shade_new_era(ax)
    ax.axhline(1, color=MUTED, lw=0.6, zorder=1)
    x, y = ratio("hay")
    ax.plot(x, y, marker="o", color=INK2, ms=3.0)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("CDL area / survey area")
    ax.set_xlabel("Year")
    _year_axis(ax)
    ax.set_xlim(2007.5, 2025.9)
    panel_label(ax, "c", "Hay (harvested area)")
    fig.tight_layout(w_pad=1.6)
    save(fig, f"fig4_rotation_area_{scope}", FIG)
    pd.DataFrame(support).to_csv(TAB / f"fig4_support_{scope}.csv", index=False)


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    setup()
    fig1(scope)
    fig2(scope)
    fig3()
    fig4(scope)
    print("figuras en", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
