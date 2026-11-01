"""Métricas de cambio a partir de tablas de transición (zone, from, to, n).

Todas las tasas se calculan sobre píxeles con dato en ambos años. Métricas por zona:
  any_change      fracción de píxeles con clase distinta (incluye rotación)
  group_change    fracción que cambia de grupo de cobertura (cdl.GROUPS)
  crop_gross      (cultivo→no cultivo + no cultivo→cultivo) / cultivo en el año inicial
  crop_net        (cultivo_final − cultivo_inicial) / cultivo inicial
  crop_type_change  entre píxeles cultivo en ambos años, fracción con clase distinta
  nonag_change    entre píxeles de bosque/urbano/agua en el año inicial, fracción que cambia de grupo
  corn_corn       maíz→maíz / maíz inicial
  cs_alternation  (maíz→soja + soja→maíz) / (maíz + soja iniciales)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agrors.cdl import AREA_CODES, AREA_CODES_SINGLE, CORN, GROUP_ORDER, SOYBEANS, UNGROUPED, group_lut

LUT = group_lut()
CROP = GROUP_ORDER.index("crop")
NONAG = [GROUP_ORDER.index(g) for g in ("forest_shrub", "developed", "water")]
METRICS = [
    "any_change",
    "group_change",
    "crop_gross",
    "crop_net",
    "crop_type_change",
    "nonag_change",
    "corn_corn",
    "cs_alternation",
]


def components(t: pd.DataFrame, by: list[str] | None = None, lut: np.ndarray | None = None) -> pd.DataFrame:
    """Sumas (numeradores y denominadores) por zona u otra clave; aditivas entre zonas.

    `lut` (código → grupo) permite recalcular con otra tabla de clases (análisis de sensibilidad).
    Las transiciones con un código sin grupo (p. ej. 81, nubes/sin dato) se descartan.
    """
    by = by or ["zone"]
    lut = LUT if lut is None else lut
    t = t[(lut[t["from"].to_numpy()] != UNGROUPED) & (lut[t["to"].to_numpy()] != UNGROUPED)]
    f = t["from"].to_numpy()
    g = t["to"].to_numpy()
    gf, gt = lut[f], lut[g]
    n = t["n"].to_numpy().astype(np.float64)
    cols = {
        "n": n,
        "changed": n * (f != g),
        "group_changed": n * (gf != gt),
        "crop0": n * (gf == CROP),
        "crop1": n * (gt == CROP),
        "crop_loss": n * ((gf == CROP) & (gt != CROP)),
        "crop_gain": n * ((gf != CROP) & (gt == CROP)),
        "crop_both": n * ((gf == CROP) & (gt == CROP)),
        "crop_both_changed": n * ((gf == CROP) & (gt == CROP) & (f != g)),
        "nonag0": n * np.isin(gf, NONAG),
        "nonag_changed": n * (np.isin(gf, NONAG) & (gf != gt)),
        "corn0": n * (f == CORN),
        "corn_corn": n * ((f == CORN) & (g == CORN)),
        "soy0": n * (f == SOYBEANS),
        "cs_alt": n * (((f == CORN) & (g == SOYBEANS)) | ((f == SOYBEANS) & (g == CORN))),
    }
    for name, codes in AREA_CODES.items():
        cols[f"area0_{name}"] = n * np.isin(f, codes)
        cols[f"area1_{name}"] = n * np.isin(g, codes)
    for name, codes in AREA_CODES_SINGLE.items():
        cols[f"area0_{name}_single"] = n * np.isin(f, codes)
        cols[f"area1_{name}_single"] = n * np.isin(g, codes)
    df = pd.DataFrame(cols)
    for k in by:
        df[k] = t[k].to_numpy()
    return df.groupby(by, as_index=False).sum()


def rates(c: pd.DataFrame) -> pd.DataFrame:
    """Tasas a partir de los componentes (cualquier nivel de agregación)."""
    out = c.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["any_change"] = c.changed / c.n
        out["group_change"] = c.group_changed / c.n
        out["crop_gross"] = (c.crop_loss + c.crop_gain) / c.crop0
        out["crop_net"] = (c.crop1 - c.crop0) / c.crop0
        out["crop_type_change"] = c.crop_both_changed / c.crop_both
        out["nonag_change"] = c.nonag_changed / c.nonag0
        out["corn_corn"] = c.corn_corn / c.corn0
        out["cs_alternation"] = c.cs_alt / (c.corn0 + c.soy0)
    return out


def excess(r_break: pd.Series, r_before: pd.Series, r_after: pd.Series) -> pd.Series:
    """Estimador del artefacto: cambio(ruptura) − media(cambio(antes), cambio(después))."""
    return r_break - (r_before + r_after) / 2


def bootstrap_ratio_stats(
    comp: dict[str, pd.DataFrame], num: str, den: str, reps: int, seed: int, clusters: pd.Series | None = None
) -> dict[str, np.ndarray]:
    """Bootstrap por condados de tres estimadores sobre tasas agregadas (Σnum/Σden):

    excess = ruptura − media(antes, después)
    step   = ruptura − antes                   (salto respecto a la línea base de la era antigua)
    persist = después − antes                  (inestabilidad que persiste en la era nueva)

    Si se da `clusters` (zona → grupo, p. ej. estado), se remuestrean grupos completos (bootstrap por
    conglomerados, más conservador ante autocorrelación espacial entre condados vecinos).
    """
    keys = ["before", "break", "after"]
    common = comp[keys[0]].index
    for k in keys[1:]:
        common = common.intersection(comp[k].index)
    arr = {k: comp[k].loc[common, [num, den]].to_numpy() for k in keys}
    rng = np.random.default_rng(seed)
    m = len(common)
    if clusters is not None:
        lab = clusters.reindex(common).to_numpy()
        groups = [np.nonzero(lab == g)[0] for g in pd.unique(lab)]
    out = {k: np.empty(reps) for k in ("excess", "step", "persist")}
    for i in range(reps):
        if clusters is None:
            idx = rng.integers(0, m, m)
        else:
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        r = {k: arr[k][idx, 0].sum() / arr[k][idx, 1].sum() for k in keys}
        out["excess"][i] = r["break"] - (r["before"] + r["after"]) / 2
        out["step"][i] = r["break"] - r["before"]
        out["persist"][i] = r["after"] - r["before"]
    return out


def bootstrap_ratio_excess(
    comp: dict[str, pd.DataFrame], num: str, den: str, reps: int, seed: int, weights_from: str | None = None
) -> np.ndarray:
    """Bootstrap por condados del exceso agregado (Σnum/Σden) entre tres pares.

    `comp` = {"before": df, "break": df, "after": df}, cada uno indexado por zona con las
    columnas `num` y `den`. Solo se usan zonas presentes en los tres. Devuelve `reps` réplicas.
    """
    keys = ["before", "break", "after"]
    common = comp[keys[0]].index
    for k in keys[1:]:
        common = common.intersection(comp[k].index)
    arr = {k: comp[k].loc[common, [num, den]].to_numpy() for k in keys}
    rng = np.random.default_rng(seed)
    m = len(common)
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, m, m)
        r = {k: arr[k][idx, 0].sum() / arr[k][idx, 1].sum() for k in keys}
        out[i] = r["break"] - (r["before"] + r["after"]) / 2
    return out
