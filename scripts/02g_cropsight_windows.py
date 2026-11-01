"""Fase 2g (E4): prepara CropSight-US y propone ventanas HLS donde es más denso.

1. data/interim/cropsight/cropsight.parquet: objetos 2021–2023 (EPSG:5070) con la clase del modelo
   E4 (corn, soybeans, hay_alfalfa, other_crop); 'cereal' se excluye por ambiguo
   (trigo de invierno/primavera, cebada, avena…).
2. Para cada región (Corn Belt: IA/IL/NE/KS; California) busca la celda de 15 km con más objetos
   2021–2023 y la traduce a tesela MGRS + desplazamiento de ventana HLS de 512 px.
   Resultado: results/tables/e4_window_candidates.csv (las ventanas elegidas se copian a config a mano).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.io import ROOT  # noqa: E402

LABEL_TO_CLASS = {
    "corn": 0,
    "soybean": 1,
    "alfalfa": 3,
    "cotton": 5,
    "almond": 5,
    "grape": 5,
    "sorghum": 5,
    "walnut": 5,
    "pistachio": 5,
    "orange": 5,
    "peanuts": 5,
    "potatoes": 5,
    "sugarbeet": 5,
    "sugarcane": 5,
    "canola": 5,
    "sunflower": 5,
}
CELL = 15360.0  # 512 px de 30 m
REGIONS = {"cornbelt": ["IA", "IL", "NE", "KS"], "california": ["CA"], "pennsylvania": ["PA"]}


def main() -> int:
    g = gpd.read_file(f"zip://{(ROOT / 'data/raw/cropsight/cropsight-us_app_dat_v1.0.1.zip').as_posix()}")
    g = g[g.Year.between(2021, 2023)].copy()
    g["model_class"] = g.Label.map(LABEL_TO_CLASS).fillna(255).astype("uint8")
    g = g.to_crs(5070)
    out = ROOT / "data" / "interim" / "cropsight"
    out.mkdir(parents=True, exist_ok=True)
    g[["Label", "Year", "Confidence", "model_class", "geometry"]].to_parquet(out / "cropsight.parquet")
    st = gpd.read_file(ROOT / "data" / "interim" / "zones" / "counties_5070.gpkg")[["STUSPS", "geometry"]]
    pts = g.set_geometry(g.representative_point())
    pts = gpd.sjoin(pts, st, predicate="within")
    pts = pts[pts.model_class != 255]
    rows = []
    for region, states in REGIONS.items():
        p = pts[pts.STUSPS.isin(states)]
        if p.empty:
            continue
        cx = np.floor(p.geometry.x / CELL)
        cy = np.floor(p.geometry.y / CELL)
        cnt = p.groupby([cx.rename("cx"), cy.rename("cy")]).size().sort_values(ascending=False)
        for (i, j), n in cnt.head(3).items():
            x, y = (i + 0.5) * CELL, (j + 0.5) * CELL
            lon, lat = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=5070).to_crs(4326).iloc[0].coords[0]
            sub = p[(cx == i) & (cy == j)]
            rows.append(
                {
                    "region": region,
                    "n_objects": int(n),
                    "center_x": x,
                    "center_y": y,
                    "lon": lon,
                    "lat": lat,
                    "state": sub.STUSPS.mode().iloc[0],
                    "years": sub.Year.value_counts().sort_index().to_dict(),
                    "labels": sub.Label.value_counts().to_dict(),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "e4_window_candidates.csv", index=False)
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
