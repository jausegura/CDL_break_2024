"""Fase 3 (E1/E3): tablas de transición por condado (zone, from, to, n) para cada par de años.

Pares: años consecutivos de config `full.years` (variante nass30) y, para los pares que tocan
la era nueva (2023→24, 2024→25), también las variantes maj30 y native10.
Salida: data/processed/transitions/{y0}_{y1}_{variante}.parquet (n en píxeles de la rejilla
de la variante: 900 m² en 30 m, 100 m² en native10).

Uso: python scripts/03a_transitions.py [pilot|full]
  pilot: solo bloques que intersectan los estados del piloto y años del piloto.
"""

from __future__ import annotations

import atexit
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import merge_counts, transition_counts  # noqa: E402
from agrors.grid import bounded_map, cdl10_is_national, cdl_path, county_path, maj30_path, nn30_path, windows  # noqa: E402
from agrors.io import ROOT, load_config  # noqa: E402
from agrors.layers import NEW_ERA_START, Layers  # noqa: E402

OUT = ROOT / "data" / "processed" / "transitions"
STRIP = 512  # filas de 30 m por franja en native10
_L: Layers | None = None
_ALLOWED: np.ndarray | None = None


def _init(allowed_zones):
    global _L, _ALLOWED
    _L = Layers()
    atexit.register(_L.close)
    _ALLOWED = allowed_zones


def _block(args):
    y0, y1, variant, win = args
    if variant == "native10" and win.height > STRIP:
        # a 10 m el bloque se procesa en franjas para limitar la memoria (máquina compartida)
        parts = []
        for r in range(0, int(win.height), STRIP):
            sub = Window(win.col_off, win.row_off + r, win.width, min(STRIP, int(win.height) - r))
            parts.append(_block((y0, y1, variant, sub)))
        return merge_counts(parts)
    grid = 10 if variant == "native10" else 30
    z30 = _L.zone(win)
    if _ALLOWED is not None:
        z30 = np.where(np.isin(z30, _ALLOWED), z30, 0).astype(z30.dtype)
    if not z30.any():
        return np.empty((0, 4), dtype=np.int64)
    z = np.repeat(np.repeat(z30, 3, 0), 3, 1) if grid == 10 else z30
    a = _L.cdl(y0, variant, win)
    b = _L.cdl(y1, variant, win)
    return transition_counts(z, a, b)


def pairs(cfg: dict, scope: str) -> list[tuple[int, int, str]]:
    if scope == "pilot":
        years = cfg["pilot"]["years"]
    else:
        y0, y1 = cfg["full"]["years"]
        years = list(range(y0, y1 + 1))
    out = []
    for a, b in zip(years[:-1], years[1:], strict=True):
        out.append((a, b, "nass30"))
        if b >= NEW_ERA_START:
            out += [(a, b, "maj30"), (a, b, "native10"), (a, b, "nn30")]
    return out


def available(y0: int, y1: int, variant: str, scope: str) -> bool:
    if scope == "full" and variant != "nass30" and not cdl10_is_national():
        return False  # el 10 m solo cubre el piloto (scripts/01d): no hay variantes nacionales
    need = []
    for y in (y0, y1):
        if y >= NEW_ERA_START and variant == "maj30":
            need.append(maj30_path(y))
        elif y == NEW_ERA_START and variant == "nn30":
            need.append(nn30_path(y))
        elif y >= NEW_ERA_START and variant == "native10":
            need.append(cdl_path(y, 10))
        else:
            need.append(cdl_path(y, 30))
    return all(p.exists() for p in need)


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    cfg = load_config()
    n_workers = cfg["compute"]["n_workers"]
    OUT.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv(ROOT / "data" / "processed" / "counties.csv", dtype={"STATEFP": str})
    allowed = None
    if scope == "pilot":
        allowed = counties.loc[counties.STUSPS.isin(cfg["pilot"]["states"]), "zone"].to_numpy()
    with rasterio.open(county_path()) as z:
        wins = list(windows(z.width, z.height, cfg["compute"]["block_size"]))
        # Descarta de antemano los bloques sin condados (océano, Canadá, México) o fuera del piloto.
        keep = []
        for w in wins:
            arr = z.read(1, window=w, out_shape=(int(w.height) // 4 + 1, int(w.width) // 4 + 1))
            if arr.any() and (allowed is None or np.isin(arr, allowed).any()):
                keep.append(w)
    print(f"{len(keep)} bloques de {len(wins)}", flush=True)
    for y0, y1, variant in pairs(cfg, scope):
        suffix = "" if scope == "full" else "_pilot"
        dest = OUT / f"{y0}_{y1}_{variant}{suffix}.parquet"
        if dest.exists() or not available(y0, y1, variant, scope):
            continue
        t0 = time.time()
        with ProcessPoolExecutor(cfg["compute"]["n_workers"], initializer=_init, initargs=(allowed,)) as ex:
            parts = list(bounded_map(ex, _block, [(y0, y1, variant, w) for w in keep], 2 * n_workers))
        m = merge_counts(parts)
        df = pd.DataFrame(m, columns=["zone", "from", "to", "n"]).astype(
            {"zone": "uint16", "from": "uint8", "to": "uint8", "n": "int64"}
        )
        df.to_parquet(dest, index=False)
        print(f"{dest.name}: {len(df)} filas, {df.n.sum():,} píxeles, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
