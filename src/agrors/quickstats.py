"""Parsing de CSV de NASS Quick Stats a tabla ordenada (cultivo, año, condado/estado, acres)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ACRE_M2 = 4046.8564224
PIXEL30_M2 = 900.0

SHORT_DESC = {
    ("corn", "planted"): "CORN - ACRES PLANTED",
    ("corn", "harvested"): "CORN, GRAIN - ACRES HARVESTED",
    ("soybeans", "planted"): "SOYBEANS - ACRES PLANTED",
    ("soybeans", "harvested"): "SOYBEANS - ACRES HARVESTED",
    ("winter_wheat", "planted"): "WHEAT, WINTER - ACRES PLANTED",
    ("winter_wheat", "harvested"): "WHEAT, WINTER - ACRES HARVESTED",
    # heno (incluye alfalfa): Quick Stats solo publica superficie cosechada
    ("hay", "harvested"): "HAY - ACRES HARVESTED",
    # algodón: a nivel de condado solo hay upland y pima por separado → se suman
    ("cotton", "planted"): ("COTTON, UPLAND - ACRES PLANTED", "COTTON, PIMA - ACRES PLANTED"),
    ("cotton", "harvested"): ("COTTON, UPLAND - ACRES HARVESTED", "COTTON, PIMA - ACRES HARVESTED"),
}


def parse_value(v: pd.Series) -> pd.Series:
    """'1,234' → 1234.0; códigos de supresión ('(D)', '(Z)', '(NA)') → NaN."""
    s = v.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.where(~s.str.startswith("(")), errors="coerce")


def tidy(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Filtra los data items de SHORT_DESC, dominio TOTAL y periodo anual; añade GEOID.

    Si un cultivo tiene varios data items (algodón upland + pima) se suman; si todos están
    suprimidos, el resultado es NaN.
    """
    inv = {}
    for k, v in SHORT_DESC.items():
        for desc in v if isinstance(v, tuple) else (v,):
            inv[desc] = k
    d = df[
        df["short_desc"].isin(inv) & (df["domain_desc"] == "TOTAL") & (df["reference_period_desc"] == "YEAR")
    ].copy()
    d["statefp"] = d["state_fips_code"].astype(str).str.zfill(2)
    if level == "county":
        d = d[d["county_ansi"].notna() & (d["county_ansi"].astype(str).str.strip() != "")]
        cnty = d["county_ansi"].astype(str).str.split(".").str[0].str.zfill(3).astype(str)
        d["GEOID"] = d["statefp"].astype(str) + cnty
    else:
        d["GEOID"] = d["statefp"]
    key = d["short_desc"].map(inv)
    d["crop"] = key.str[0]
    d["stat"] = key.str[1]
    d["acres"] = parse_value(d["Value"])
    d["year"] = d["year"].astype(int)
    out = (
        d.groupby(["GEOID", "statefp", "year", "crop", "stat"])["acres"]
        .sum(min_count=1)
        .unstack("stat")
        .reset_index()
    )
    out.columns.name = None
    for c in ("planted", "harvested"):
        if c not in out:
            out[c] = np.nan
    return out[["GEOID", "statefp", "year", "crop", "planted", "harvested"]]


def acres_to_pixels30(acres: pd.Series | float) -> pd.Series | float:
    return acres * ACRE_M2 / PIXEL30_M2
