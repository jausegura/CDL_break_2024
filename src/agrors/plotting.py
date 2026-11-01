"""Estilo editorial de las figuras (formato de revista, Elsevier).

- Anchos de columna Elsevier: simple 90 mm, 1,5 columnas 140 mm, doble 190 mm.
- Tipografía Source Sans 3 (OFL, paper/fonts/), 7–8 pt a tamaño final; letras de panel en negrita.
- Paleta categórica comprobada para todos los pares de colores, también con daltonismo:
  azul acero, terracota y verde azulado, más gris neutro para series de contexto.
- Exportación: PDF vectorial con fuentes incrustadas (TrueType) y PNG a 600 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from agrors.io import ROOT

MM = 1 / 25.4
WIDTH_DOUBLE = 190 * MM
WIDTH_ONEHALF = 140 * MM
WIDTH_SINGLE = 90 * MM

BLUE, TERRACOTTA, TEAL = "#2f63a3", "#c65a3c", "#1d9277"
OCHRE, OCHRE_DARK = (
    "#d4a020",
    "#8f6a0f",
)  # 4.º color (validado con los tres anteriores); borde oscuro por contraste
SERIES = [BLUE, TERRACOTTA, TEAL]
GREY = "#8a8f98"
INK, INK2, MUTED = "#1d1f23", "#4a4f57", "#7d828a"
GRID, AXIS = "#e4e4e1", "#3a3d42"
ERA_FILL = "#f2ede6"  # sombreado del periodo del método nuevo (2024–2025)

# Versiones alternativas del mapa nuevo: siempre círculos; se distinguen por color y por relleno.
VARIANT_STYLE = {
    "maj30": {"marker": "o", "color": TERRACOTTA, "mec": "#8e3423", "mew": 0.7},
    "native10": {
        "marker": "o",
        "color": TEAL,
        "fillstyle": "bottom",
        "mfcalt": "white",
        "mec": TEAL,
        "mew": 0.9,
    },
    "nn30": {"marker": "o", "color": OCHRE, "mfc": OCHRE, "mec": OCHRE_DARK, "mew": 0.9},
}
VARIANT_LABEL = {
    "nass30": "NASS 30 m product",
    "nn30": "Nearest neighbour of 10 m",
    "maj30": "3 × 3 majority of 10 m",
    "native10": "Native 10 m",
}
# Mapa: divergente con punto medio neutro cálido; el lado positivo es una rampa monótona en luminosidad.
DIVERGING = LinearSegmentedColormap.from_list(
    "break_excess",
    ["#2f63a3", "#9fb8d6", "#f4f1ec", "#efcdb9", "#e09a7c", "#c65a3c", "#8e3423", "#56190f"],
)


def _register_fonts() -> str:
    family = "DejaVu Sans"
    for f in sorted((ROOT / "paper" / "fonts").glob("SourceSans3-*.ttf")):
        font_manager.fontManager.addfont(str(f))
        family = "Source Sans 3"
    return family


def setup() -> None:
    family = _register_fonts()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "regular",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.6,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
        }
    )


def panel_label(ax, letter: str, title: str = "", x: float = -0.02, y: float = 1.06) -> None:
    """Letra de panel en negrita y título breve en la misma línea, alineados a la izquierda."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="right",
        va="bottom",
    )
    if title:
        ax.text(x + 0.02, y, title, transform=ax.transAxes, fontsize=8, color=INK, ha="left", va="bottom")


def shade_new_era(ax, start: float = 2023.5, end: float = 2025.8, label: bool = False) -> None:
    """Sombrea el periodo del método nuevo y marca la ruptura con una línea fina."""
    ax.axvspan(start, end, color=ERA_FILL, zorder=0, lw=0)
    ax.axvline(start, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    if label:
        ax.text(
            start + 0.15,
            0.04,
            "RF\n10 m",
            transform=ax.get_xaxis_transform(),
            fontsize=6.3,
            color=INK2,
            va="bottom",
            ha="left",
            linespacing=1.05,
        )
        ax.text(
            start - 0.2,
            0.04,
            "See5, 30 m",
            transform=ax.get_xaxis_transform(),
            fontsize=6.3,
            color=INK2,
            va="bottom",
            ha="right",
        )


def end_label(ax, x, y, text: str, color: str, dy: float = 0.0, dx: float = 0.25) -> None:
    """Etiqueta directa al final de una serie (sustituye a la leyenda)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=6.8,
        color=color,
        va="center",
        ha="left",
        annotation_clip=False,
    )


def save(fig, name: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", 600), ("png", 600)):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    plt.close(fig)
    return paths


def end_labels(ax, x: float, items: list[tuple[float, str, str]], min_gap: float, dx: float = 0.3) -> None:
    """Etiquetas directas al final de varias series, separadas al menos `min_gap` (unidades de datos)
    para que no se solapen; se mantiene el orden vertical de las series."""
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = (sum(y for y, _, _ in items) - sum(ys)) / len(ys)  # recentra el bloque de etiquetas
    for (_, text, color), y in zip(items, ys, strict=True):
        ax.annotate(
            text,
            (x + dx, y + shift),
            fontsize=6.8,
            color=color,
            va="center",
            ha="left",
            annotation_clip=False,
        )
