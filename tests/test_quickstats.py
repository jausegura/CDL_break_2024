import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.quickstats import acres_to_pixels30, parse_value, tidy  # noqa: E402


def _row(short, county, value, domain="TOTAL"):
    return {
        "short_desc": short,
        "domain_desc": domain,
        "reference_period_desc": "YEAR",
        "state_fips_code": "19",
        "county_ansi": county,
        "Value": value,
        "year": "2023",
    }


def test_parse_value():
    s = parse_value(pd.Series(["1,234", " (D)", "56", "(Z)"]))
    assert s.iloc[0] == 1234 and pd.isna(s.iloc[1]) and s.iloc[2] == 56 and pd.isna(s.iloc[3])


def test_tidy_county_filters_and_geoid():
    df = pd.DataFrame(
        [
            _row("CORN - ACRES PLANTED", "1", "1,000"),
            _row("CORN, GRAIN - ACRES HARVESTED", "1", "900"),
            _row("CORN - ACRES PLANTED", None, "50"),  # OTHER (COMBINED) COUNTIES
            _row("CORN, SILAGE - ACRES HARVESTED", "1", "70"),  # no es un data item objetivo
            _row("CORN - ACRES PLANTED", "3", "(D)"),
            _row("CORN - ACRES PLANTED", "5", "10", domain="IRRIGATION"),
        ]
    )
    t = tidy(df, "county")
    assert set(t.GEOID) == {"19001", "19003"}
    r = t[t.GEOID == "19001"].iloc[0]
    assert r.planted == 1000 and r.harvested == 900 and r.crop == "corn"
    assert pd.isna(t[t.GEOID == "19003"].iloc[0].planted)


def test_acres_to_pixels():
    assert abs(acres_to_pixels30(900 / 4046.8564224) - 1.0) < 1e-12


def test_tidy_cotton_sums_upland_and_pima():
    df = pd.DataFrame(
        [
            _row("COTTON, UPLAND - ACRES PLANTED", "1", "100"),
            _row("COTTON, PIMA - ACRES PLANTED", "1", "20"),
            _row("COTTON, UPLAND - ACRES PLANTED", "3", "(D)"),
        ]
    )
    t = tidy(df, "county").set_index("GEOID")
    assert t.loc["19001", "planted"] == 120 and pd.isna(t.loc["19003", "planted"])
