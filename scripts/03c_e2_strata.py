"""Fase 3 (E2/E5): cambio de clase estratificado por tamaño de parcela CSB y distancia al borde.

Geometría de referencia: CSB 2016–2023 (construido solo con CDL de la era antigua). Los polígonos
se rasterizan a 10 m; la distancia al borde se calcula a 10 m (EDT) y cada píxel de 30 m toma el
id y la distancia de su subpíxel central. Para cada par de años y variante se cuentan, por
estrato (estado, decil de tamaño, banda de borde, clase de origen), los píxeles válidos, los que
cambian de clase y los que pasan de cultivo a no cultivo o viceversa.

Salida: results/tables/e2_strata_pilot.csv y results/tables/e2_size_deciles.csv
"""

from __future__ import annotations

import atexit
import hashlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import (  # noqa: E402
    CORN,
    COTTON,
    GROUP_ORDER,
    SOYBEANS,
    UNGROUPED,
    WINTER_WHEAT,
    center3,
    group_lut,
)
from agrors.fields import edge_bin, edge_distance_m, size_deciles  # noqa: E402
from agrors.grid import bounded_map, county_path, windows  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.layers import Layers  # noqa: E402

PAD = 8  # píxeles de 30 m de margen (240 m) para que el borde del bloque no sea un borde falso
BLOCK = 1024
CLASS_NAMES = ["corn", "soybeans", "winter_wheat", "cotton", "other_crop", "noncrop"]
LUT = group_lut()
CROP = GROUP_ORDER.index("crop")
STATES = {"IA": 19, "IL": 17, "NE": 31, "KS": 20, "PA": 42, "CA": 6}
_G = _L = _DEC = _STATE_OF_ZONE = None


def from_class(a: np.ndarray) -> np.ndarray:
    c = np.full(a.shape, 5, dtype=np.uint8)
    c[LUT[a] == CROP] = 4
    for k, code in enumerate((CORN, SOYBEANS, WINTER_WHEAT, COTTON)):
        c[a == code] = k
    return c


def _init(gpkg_path, dec_path):
    global _G, _L, _DEC, _STATE_OF_ZONE
    _G = gpd.read_parquet(gpkg_path)
    _DEC = np.load(dec_path)
    _L = Layers()
    atexit.register(_L.close)
    c = pd.read_csv(ROOT / "data" / "processed" / "counties.csv", dtype={"STATEFP": str})
    lut = np.zeros(int(c.zone.max()) + 1, dtype=np.uint8)
    lut[c.zone.values] = c.STATEFP.astype(int).values
    _STATE_OF_ZONE = lut


def _block(args):
    win, pairs = args
    wp = Window(win.col_off - PAD, win.row_off - PAD, win.width + 2 * PAD, win.height + 2 * PAD)
    t10 = win_transform(Window(wp.col_off * 3, wp.row_off * 3, wp.width * 3, wp.height * 3), _L.t10)
    h10, w10 = int(wp.height) * 3, int(wp.width) * 3
    x0, y0 = t10.c, t10.f
    sub = _G.iloc[list(_G.sindex.intersection((x0, y0 + t10.e * h10, x0 + t10.a * w10, y0)))]
    if sub.empty:
        return []
    ids10 = rasterize(
        zip(sub.geometry, sub.gid, strict=True), out_shape=(h10, w10), transform=t10, fill=0, dtype="int32"
    )
    eb10 = edge_bin(edge_distance_m(ids10, 10.0))
    s = PAD * 3
    ids10, eb10 = ids10[s:-s, s:-s], eb10[s:-s, s:-s]
    ids30, eb30 = center3(ids10), center3(eb10)
    zone30 = _L.zone(win)
    st30 = _STATE_OF_ZONE[zone30]
    st10 = np.repeat(np.repeat(st30, 3, 0), 3, 1)
    out = []
    cache: dict = {}

    def get(year, variant):
        k = (year, variant)
        if k not in cache:
            cache[k] = _L.cdl(year, variant, win)
        return cache[k]

    for y0_, y1_, variant in pairs:
        if variant == "native10":
            ids, eb, st = ids10, eb10, st10
        else:
            ids, eb, st = ids30, eb30, st30
        a, b = get(y0_, variant), get(y1_, variant)
        m = (ids > 0) & (LUT[a] != UNGROUPED) & (LUT[b] != UNGROUPED) & (st > 0)
        if not m.any():
            continue
        dec = _DEC[ids[m]]
        cls = from_class(a[m])
        key = ((st[m].astype(np.int64) * 10 + dec) * 3 + eb[m]) * 6 + cls
        n = np.bincount(key, minlength=100 * 180)
        ch = np.bincount(key, weights=(a[m] != b[m]), minlength=100 * 180)
        ga, gb = LUT[a[m]] == CROP, LUT[b[m]] == CROP
        gc = np.bincount(key, weights=(ga != gb), minlength=100 * 180)
        nz = np.nonzero(n)[0]
        out.append((y0_, y1_, variant, nz, n[nz], ch[nz], gc[nz]))
    return out


def main() -> int:
    cfg = load_config()
    n_workers = cfg["compute"]["e2_workers"]
    csb_dir = ROOT / "data" / "interim" / "csb"
    states = cfg["pilot"]["states"]
    # deciles de tamaño comunes a todo el piloto (solo la columna de superficie)
    acres = np.concatenate(
        [
            pd.read_parquet(csb_dir / f"csb2016-2023_{s}.parquet", columns=["CSBACRES"]).CSBACRES.to_numpy()
            for s in states
        ]
    )
    _, cuts = size_deciles(acres)
    pd.DataFrame(
        {
            "decile": range(10),
            "acres_low": np.r_[acres.min(), cuts],
            "acres_high": np.r_[cuts, acres.max()],
            "n_polygons": np.bincount(np.digitize(acres, cuts), minlength=10),
        }
    ).to_csv(ROOT / "results" / "tables" / "e2_size_deciles.csv", index=False)
    del acres
    years = cfg["pilot"]["years"]
    pairs = []
    for a, b in zip(years[:-1], years[1:], strict=True):
        pairs.append((a, b, "nass30"))
        if b >= 2024:
            pairs += [(a, b, "maj30"), (a, b, "native10")]
    with rasterio.open(county_path()) as z:
        t = z.transform
        allw = list(windows(z.width, z.height, BLOCK))
    # resultados parciales por estado: si el proceso se interrumpe, se reanuda desde el último estado
    # completo; la clave cambia si cambian la tabla de clases o los pares
    tag = hashlib.sha256(LUT.tobytes() + repr(pairs).encode()).hexdigest()[:12]
    parts = ROOT / "data" / "interim" / "e2_parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for state in states:
        part = parts / f"{state}_{tag}.pkl"
        if part.exists():
            print(state, "ya calculado", flush=True)
            continue
        acc: dict = {}
        g = gpd.read_parquet(csb_dir / f"csb2016-2023_{state}.parquet", columns=["CSBACRES", "geometry"])
        g["gid"] = np.arange(1, len(g) + 1, dtype=np.int32)
        dec_lut = np.zeros(len(g) + 1, dtype=np.uint8)
        dec_lut[1:] = np.digitize(g.CSBACRES.to_numpy(), cuts)
        tmp = csb_dir / "_e2_polys.parquet"
        g[["gid", "geometry"]].to_parquet(tmp)
        dec_path = csb_dir / "_e2_deciles.npy"
        np.save(dec_path, dec_lut)
        xmin, ymin, xmax, ymax = g.total_bounds
        sidx = g.sindex
        keep = []
        for w in allw:
            wt = win_transform(w, t)
            bb = (wt.c, wt.f + wt.e * w.height, wt.c + wt.a * w.width, wt.f)
            if bb[2] < xmin or bb[0] > xmax or bb[3] < ymin or bb[1] > ymax:
                continue
            if len(sidx.intersection(bb)):
                keep.append(w)
        print(state, len(keep), "bloques;", len(g), "polígonos", flush=True)
        del g, sidx
        with ProcessPoolExecutor(n_workers, initializer=_init, initargs=(str(tmp), str(dec_path))) as ex:
            for i, res in enumerate(bounded_map(ex, _block, [(w, pairs) for w in keep], 2 * n_workers)):
                for y0_, y1_, variant, nz, n, ch, gc in res:
                    d = acc.setdefault((y0_, y1_, variant), {})
                    for k, a1, a2, a3 in zip(nz, n, ch, gc, strict=True):
                        v = d.get(k)
                        d[k] = (a1, a2, a3) if v is None else (v[0] + a1, v[1] + a2, v[2] + a3)
                if i % 50 == 0:
                    print(f"  {state} {i}/{len(keep)} {time.time() - t0:.0f} s", flush=True)
        with open(part.with_suffix(".tmp"), "wb") as f:
            pickle.dump(acc, f)
        part.with_suffix(".tmp").replace(part)
    acc = {}
    for state in states:
        with open(parts / f"{state}_{tag}.pkl", "rb") as f:
            for key, d in pickle.load(f).items():
                tot = acc.setdefault(key, {})
                for k, v in d.items():
                    w = tot.get(k)
                    tot[k] = v if w is None else (w[0] + v[0], w[1] + v[1], w[2] + v[2])
    rows = []
    fips_to_st = {v: k for k, v in STATES.items()}
    for (y0_, y1_, variant), d in acc.items():
        for k, (n, ch, gc) in d.items():
            cls = k % 6
            eb = (k // 6) % 3
            dec_ = (k // 18) % 10
            st = k // 180
            rows.append(
                {
                    "state": fips_to_st.get(int(st), str(st)),
                    "y0": y0_,
                    "y1": y1_,
                    "variant": variant,
                    "size_decile": int(dec_),
                    "edge_bin": int(eb),
                    "from_class": CLASS_NAMES[cls],
                    "n": n,
                    "changed": ch,
                    "crop_noncrop_changed": gc,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "e2_strata_pilot.csv", index=False)
    print("escrito", len(out), "filas", f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
