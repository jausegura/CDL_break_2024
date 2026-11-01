"""Fase 3 (E4): propagación de la ruptura a modelos entrenados con etiquetas CDL.

LightGBM sobre medianas mensuales HLS (mayo–sept., 6 bandas + NDVI). Partición espacial por
bloques de 1,5 km (70 % entrenamiento / 30 % test, semilla fija), común a todos los años.
Conjuntos de entrenamiento (mismo tamaño de muestra, estratificado con tope por clase):
  OLD3   2021–2023 (etiquetas CDL de la era antigua)
  OLD1   2023
  OLD2   2021–2022 (para evaluar 2023 fuera de año)
  NEW    2024, etiquetas NASS 30 m (vecino más próximo del 10 m)
  NEWmaj 2024, etiquetas mayoría 3x3 desde 10 m
Conjuntos de test (bloques de test):
  T2025_cdl        2025 frente a CDL 2025 (era nueva)
  T2025_cdl_int    idem, solo píxeles a ≥60 m del borde de la parcela CSB 2016–23
  T2023_cdl        2023 frente a CDL 2023 (era antigua; solo modelos que no usan 2023)
  T20xx_cropsight  2021, 2022, 2023 y 2021+22 frente a CropSight-US (referencia independiente);
                   cada modelo solo se evalúa en años en que no se entrenó; frente a CropSight se
                   informa también la exactitud por objeto (voto mayoritario) y un IC con bootstrap
                   de objetos
Salida: results/tables/e4_metrics.csv (una fila por modelo × test × semilla).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import GROUP_ORDER, UNGROUPED, group_lut  # noqa: E402
from agrors.fields import edge_distance_m  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402

HLS = ROOT / "data" / "processed" / "hls"
CLASSES = ["corn", "soybeans", "winter_wheat", "hay_alfalfa", "grass_pasture", "other_crop", "non_ag"]
LUT = group_lut()
BLOCK_PX = 50  # 1,5 km


def model_class(cdl: np.ndarray) -> np.ndarray:
    """Códigos CDL → índice de CLASSES; 255 = fondo/sin dato."""
    out = np.full(cdl.shape, 255, dtype=np.uint8)
    g = LUT[cdl]
    out[g != UNGROUPED] = 6
    out[g == GROUP_ORDER.index("crop")] = 5
    out[g == GROUP_ORDER.index("grass_pasture")] = 4
    out[np.isin(cdl, (36, 37))] = 3
    out[np.isin(cdl, (24, 26))] = 2
    out[cdl == 5] = 1
    out[cdl == 1] = 0
    return out


def block_split(shape, seed: int) -> np.ndarray:
    """True = test. Bloques de BLOCK_PX asignados al azar (30 % test)."""
    rng = np.random.default_rng(seed)
    nb = (shape[0] // BLOCK_PX + 1, shape[1] // BLOCK_PX + 1)
    tb = rng.random(nb) < 0.3
    return np.kron(tb, np.ones((BLOCK_PX, BLOCK_PX), dtype=bool))[: shape[0], : shape[1]]


def features(win: str, year: int) -> tuple[np.ndarray, Affine, str]:
    d = np.load(HLS / f"{win}_{year}.npz")
    X = d["X"].astype(np.float32)  # (meses, 7, h, w)
    m, b, h, w = X.shape
    return X.reshape(m * b, h * w).T, Affine(*d["transform"]), str(d["crs"])


def interior_mask(win_cfg: dict, transform: Affine, crs: str, shape) -> np.ndarray:
    """Píxeles HLS cuyo centro está a ≥60 m del borde de su polígono CSB 2016–23."""
    st = win_cfg["state"]
    g = gpd.read_parquet(ROOT / "data" / "interim" / "csb" / f"csb2016-2023_{st}.parquet").to_crs(crs)
    t10 = Affine(transform.a / 3, 0, transform.c, 0, transform.e / 3, transform.f)
    h10, w10 = shape[0] * 3, shape[1] * 3
    x0, y0 = t10.c, t10.f
    sub = g.cx[x0 : x0 + t10.a * w10, y0 + t10.e * h10 : y0]
    ids = rasterize(
        zip(sub.geometry, range(1, len(sub) + 1), strict=True),
        out_shape=(h10, w10),
        transform=t10,
        fill=0,
        dtype="int32",
    )
    d = edge_distance_m(ids, 10.0)[1::3, 1::3]
    return np.nan_to_num(d, nan=0) >= 60


def sample(y: np.ndarray, valid: np.ndarray, n: int, cap: int, rng) -> np.ndarray:
    idx = np.nonzero(valid)[0]
    out = []
    for c in range(len(CLASSES)):
        ic = idx[y[idx] == c]
        if len(ic):
            out.append(rng.choice(ic, size=min(cap, len(ic)), replace=False))
    sel = np.concatenate(out)
    if len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)
    return sel


def cropsight_labels(year: int, transform: Affine, crs: str, shape) -> tuple[np.ndarray, np.ndarray] | None:
    """Etiquetas CropSight-US de `year` rasterizadas en la ventana (255 = sin objeto) y el identificador
    del objeto de cada píxel (0 = sin objeto), para evaluar por objeto: sus píxeles no son independientes."""
    path = ROOT / "data" / "interim" / "cropsight" / "cropsight.parquet"
    if not path.exists():
        return None
    g = gpd.read_parquet(path)
    g = g[g.Year == year].to_crs(crs)
    h, w = shape
    x0, y0 = transform.c, transform.f
    sub = g.cx[x0 : x0 + transform.a * w, y0 + transform.e * h : y0]
    sub = sub[sub.model_class != 255]
    if sub.empty:
        return None
    labels = rasterize(
        zip(sub.geometry, sub.model_class, strict=True),
        out_shape=shape,
        transform=transform,
        fill=255,
        dtype="uint8",
    )
    ids = rasterize(
        zip(sub.geometry, np.arange(1, len(sub) + 1), strict=True),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return labels, ids


def object_scores(y_true, y_pred, obj, reps: int, seed: int) -> dict:
    """Evaluación por objeto CropSight: exactitud del voto mayoritario de cada objeto e IC al 95 % de la
    exactitud por píxel con bootstrap de objetos (los píxeles de un mismo objeto no son independientes)."""
    _, inv = np.unique(obj, return_inverse=True)
    n_obj = inv.max() + 1
    n = np.bincount(inv, minlength=n_obj)
    ok = np.bincount(inv, weights=(y_true == y_pred), minlength=n_obj)
    lab = np.zeros(n_obj, dtype=int)
    lab[inv] = y_true
    votes = np.zeros((n_obj, len(CLASSES)))
    np.add.at(votes, (inv, y_pred), 1)
    rng = np.random.default_rng(seed)
    boot = [ok[ix].sum() / n[ix].sum() for ix in (rng.integers(0, n_obj, n_obj) for _ in range(reps))]
    return {
        "n_objects": int(n_obj),
        "oa_object": float((votes.argmax(1) == lab).mean()),
        "oa_ci_low_obj": float(np.quantile(boot, 0.025)),
        "oa_ci_high_obj": float(np.quantile(boot, 0.975)),
    }


def scores(y_true, y_pred) -> dict:
    present = sorted(set(np.unique(y_true)))
    f1 = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    out = {"oa": accuracy_score(y_true, y_pred), "f1_macro": float(np.mean(f1)), "n_test": len(y_true)}
    for c, v in zip(present, f1, strict=True):
        out[f"f1_{CLASSES[c]}"] = v
    return out


def main() -> int:
    cfg = load_config()
    e4 = cfg["model_e4"]
    rows = []
    data = {}
    for wc in e4["windows"]:
        lab = np.load(HLS / f"{wc['name']}_labels.npz")
        for y in cfg["pilot"]["years"]:
            X, t, crs = features(wc["name"], y)
            data[(wc["name"], y)] = X
        shape = (wc["size"], wc["size"])
        data[(wc["name"], "test")] = block_split(shape, cfg["seed"]).ravel()
        data[(wc["name"], "interior")] = interior_mask(wc, t, crs, shape).ravel()
        for y in (2021, 2022, 2023):
            cs = cropsight_labels(y, t, crs, shape)
            if cs is not None:
                data[(wc["name"], f"cs_{y}")], data[(wc["name"], f"csid_{y}")] = cs
        for k in lab.files:
            data[(wc["name"], k)] = model_class(lab[k]).ravel()
    train_sets = {
        "OLD3": [(y, "nass30") for y in (2021, 2022, 2023)],
        "OLD1": [(2023, "nass30")],
        "OLD2": [(2021, "nass30"), (2022, "nass30")],
        "NEW": [(2024, "nass30")],
        "NEWmaj": [(2024, "maj30")],
    }
    n_train = e4["n_train_per_window"]
    for seed in e4["seeds"]:
        rng = np.random.default_rng(seed)
        for name, spec in train_sets.items():
            Xs, ys = [], []
            for wc in e4["windows"]:
                wn = wc["name"]
                per = n_train // len(spec)
                for y, v in spec:
                    yy = data[(wn, f"cdl_{y}_{v}")]
                    valid = (~data[(wn, "test")]) & (yy != 255)
                    sel = sample(yy, valid, per, per // 3, rng)
                    Xs.append(data[(wn, y)][sel])
                    ys.append(yy[sel])
            X, y = np.concatenate(Xs), np.concatenate(ys)
            clf = lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
            clf.fit(X, y)
            trained = {y for y, _ in spec}
            tests = {
                "T2025_cdl": [(2025, "cdl_2025_nass30")],
                "T2025_cdl_maj": [(2025, "cdl_2025_maj30")],
                "T2025_cdl_int": [(2025, "cdl_2025_nass30", "interior")],
                "T2023_cdl": [(2023, "cdl_2023_nass30")],
                "T2021_cropsight": [(2021, "cs_2021")],
                "T2022_cropsight": [(2022, "cs_2022")],
                "T2023_cropsight": [(2023, "cs_2023")],
                "T2122_cropsight": [(2021, "cs_2021"), (2022, "cs_2022")],
            }
            for tname, parts_ in tests.items():
                if trained & {pt[0] for pt in parts_}:
                    continue  # el modelo vio ese año en entrenamiento
                yt_all, yp_all, obj_all = [], [], []
                for wc_i, wc in enumerate(e4["windows"]):
                    wn = wc["name"]
                    for pt in parts_:
                        ty, key = pt[0], pt[1]
                        yt = data.get((wn, key))
                        if yt is None:
                            continue
                        yt = yt.ravel()
                        m = data[(wn, "test")] & (yt != 255)
                        if len(pt) > 2:
                            m &= data[(wn, pt[2])]
                        if not m.any():
                            continue
                        yt_all.append(yt[m])
                        yp_all.append(clf.predict(data[(wn, ty)][m]))
                        if key.startswith("cs_"):  # identificador único por ventana, año y objeto
                            ids = data[(wn, f"csid_{ty}")].ravel()[m].astype(np.int64)
                            obj_all.append((wc_i * 10_000 + ty) * 1_000_000 + ids)
                if not yt_all:
                    continue
                rows.append(
                    {
                        "train": name,
                        "test": tname,
                        "seed": seed,
                        "n_train": len(y),
                        **scores(np.concatenate(yt_all), np.concatenate(yp_all)),
                        **(
                            object_scores(
                                np.concatenate(yt_all),
                                np.concatenate(yp_all),
                                np.concatenate(obj_all),
                                cfg["stats"]["bootstrap_reps"],
                                seed,
                            )
                            if obj_all
                            else {}
                        ),
                    }
                )
                r = rows[-1]
                print(r["train"], tname, seed, round(r["oa"], 4), round(r["f1_macro"], 4), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "e4_metrics.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
