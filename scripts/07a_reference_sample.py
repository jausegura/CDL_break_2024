"""Fase 7a: diseño de una muestra de referencia estratificada para validar el cambio 2023→2024.

Estratos definidos por la transición de grupo de cobertura en el producto NASS 30 m (cdl.GROUPS):
cultivo estable, pastizal estable, otros estables, cultivo→pastizal, pastizal→cultivo,
bosque/matorral→pastizal y otros cambios. Diseño en dos etapas con probabilidades conocidas:
  1. muestreo aleatorio simple sin reemplazo de `n_tiles` teselas de la partición nacional en teselas de
     `tile_size` × `tile_size` píxeles (solo teselas con algún píxel de CONUS);
  2. muestreo aleatorio simple de `n_per_stratum[s]` píxeles entre todos los píxeles del estrato s en las
     teselas elegidas.
Cada píxel del estrato tiene así la misma probabilidad de inclusión (n_tiles/N_tiles · n_s/M_s); los
tamaños N_s de los estratos salen de las tablas de transición nacionales y permiten el estimador
estratificado de área y de exactitud de Olofsson et al. (2014) tras la fotointerpretación.
Salidas (results/reference_sample/):
  sample_2023_2024.csv   puntos con coordenadas (EPSG:5070 y WGS84), estrato, clases CDL y columnas vacías
                         para la interpretación
  strata_2023_2024.csv   N_s (píxeles CONUS), candidatos M_s en las teselas y n_s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import GROUP_ORDER, UNGROUPED, group_lut  # noqa: E402
from agrors.grid import cdl_path, read_aligned  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.layers import Layers  # noqa: E402

OUT = ROOT / "results" / "reference_sample"
G = {name: i for i, name in enumerate(GROUP_ORDER)}


def stratum(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Índice de estrato (orden de load_config()['reference_sample']['n_per_stratum']); -1 sin dato."""
    crop, grass, fs = G["crop"], G["grass_pasture"], G["forest_shrub"]
    s = np.full(g0.shape, 6, dtype=np.int8)  # other_change
    same = g0 == g1
    s[same] = 2  # stable_other
    s[same & (g0 == crop)] = 0
    s[same & (g0 == grass)] = 1
    s[(g0 == crop) & (g1 == grass)] = 3
    s[(g0 == grass) & (g1 == crop)] = 4
    s[(g0 == fs) & (g1 == grass)] = 5
    s[(g0 == UNGROUPED) | (g1 == UNGROUPED)] = -1
    return s


def population_sizes(y0: int, y1: int, lut: np.ndarray) -> np.ndarray:
    t = pd.read_parquet(ROOT / "data" / "processed" / "transitions" / f"{y0}_{y1}_nass30.parquet")
    s = stratum(lut[t["from"].to_numpy()], lut[t["to"].to_numpy()])
    ok = s >= 0
    return np.bincount(s[ok], weights=t.n.to_numpy()[ok], minlength=7)


def main() -> int:
    cfg = load_config()
    rc = cfg["reference_sample"]
    y0, y1 = rc["pair"]
    names = list(rc["n_per_stratum"])
    size = rc["tile_size"]
    lut = group_lut()
    rng = np.random.default_rng(cfg["seed"])
    L = Layers()
    nr, nc = L.shape30[0] // size, L.shape30[1] // size
    order = rng.permutation(nr * nc)
    # reservorio por estrato: se guarda el n_s de claves aleatorias más pequeñas entre todos los candidatos
    # (equivale a un muestreo aleatorio simple de los candidatos con memoria O(n_s))
    k_s = [rc["n_per_stratum"][n] for n in names]
    res = [np.empty((0, 6)) for _ in names]  # clave, fila, columna, clase y0, clase y1, zona
    m_s = np.zeros(len(names), dtype=np.int64)
    used = 0
    with rasterio.open(cdl_path(y0, 30)) as s0, rasterio.open(cdl_path(y1, 30)) as s1:
        for k in order:
            if used == rc["n_tiles"]:
                break
            r, c = divmod(int(k), nc)
            w = Window(c * size, r * size, size, size)
            z = L.zone(w)
            if not (z > 0).any():
                continue
            a = read_aligned(s0, L.t30, w, (size, size))
            b = read_aligned(s1, L.t30, w, (size, size))
            st = stratum(lut[a], lut[b])
            st[z == 0] = -1
            if not (st >= 0).any():
                continue
            used += 1
            for si in range(len(names)):
                rr, cc = np.nonzero(st == si)
                m_s[si] += len(rr)
                if not len(rr):
                    continue
                key = rng.random(len(rr))
                keep = np.argsort(key)[: k_s[si]]
                new = np.column_stack(
                    [
                        key[keep],
                        r * size + rr[keep],
                        c * size + cc[keep],
                        a[rr, cc][keep],
                        b[rr, cc][keep],
                        z[rr, cc][keep],
                    ]
                )
                both = np.vstack([res[si], new])
                res[si] = both[np.argsort(both[:, 0])[: k_s[si]]]
    L.close()
    to_wgs = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    counties = pd.read_csv(ROOT / "data" / "processed" / "counties.csv", dtype={"GEOID": str})
    geoid = counties.set_index("zone").GEOID
    rows, strata = [], []
    pop = population_sizes(y0, y1, lut)
    for si, name in enumerate(names):
        strata.append(
            {"stratum": name, "N_pixels_conus": int(pop[si]), "M_candidates": int(m_s[si]), "n": len(res[si])}
        )
        for _, row, col, ca, cb, zone in res[si]:
            row, col, ca, cb, zone = int(row), int(col), int(ca), int(cb), int(zone)
            x = L.t30.c + (col + 0.5) * L.t30.a
            y = L.t30.f + (row + 0.5) * L.t30.e
            lon, lat = to_wgs.transform(x, y)
            rows.append(
                {
                    "stratum": name,
                    "x_5070": x,
                    "y_5070": y,
                    "lon": round(lon, 6),
                    "lat": round(lat, 6),
                    "GEOID": geoid.get(zone, ""),
                    f"cdl_{y0}": ca,
                    f"cdl_{y1}": cb,
                    f"ref_group_{y0}": "",
                    f"ref_group_{y1}": "",
                    "ref_confidence": "",
                    "imagery_used": "",
                    "interpreter": "",
                    "notes": "",
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    df.insert(0, "point_id", [f"P{i + 1:04d}" for i in range(len(df))])
    df.to_csv(OUT / f"sample_{y0}_{y1}.csv", index=False)
    sd = pd.DataFrame(strata).assign(n_tiles=used, N_tiles=nr * nc, tile_size=size)
    sd.to_csv(OUT / f"strata_{y0}_{y1}.csv", index=False)
    print(sd.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
