from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from production.decline import arps_rate
from production.rrc import load_production, parse_field_leases
from production.typecurve import build_type_curve, pick_candidates

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_field(scales=(0.5, 0.8, 1.0, 1.25, 2.0), qi=400.0, di=0.12, b=1.0, months=60):
    """Wells with the same decline shape but different sizes, drilled in different years."""
    rows = []
    for i, s in enumerate(scales):
        dates = pd.date_range(f"{2012 + i}-01-01", periods=months + 2, freq="MS")
        t = np.arange(len(dates)) - 2  # two ramp-up months before the peak
        bopd = np.where(t < 0, s * qi * 0.4, s * arps_rate(np.maximum(t, 0), qi, di, b))
        rows.append(pd.DataFrame({"lease_no": 1000 + i, "lease_name": f"WELL {i}", "date": dates, "bopd": bopd,
                                  "oil_bbl": bopd * dates.days_in_month, "gas_mcf": bopd * dates.days_in_month * 2,
                                  "gas_mcfd": bopd * 2}))
    return pd.concat(rows, ignore_index=True)


def test_type_curve_recovers_shared_shape():
    tc = build_type_curve(synthetic_field())
    assert tc.n_wells == 5
    assert tc.profile["t"].iloc[0] == 0                       # aligned at each well's peak
    assert tc.profile["p50"].iloc[0] == pytest.approx(400.0)   # median well is the scale-1.0 well
    p50 = tc.fits["P50"]
    assert p50.b == pytest.approx(1.0, abs=0.05) and p50.di == pytest.approx(0.12, rel=0.05)
    # P10 is the high case, P90 the low case
    assert (tc.profile["p10"] >= tc.profile["p50"]).all() and (tc.profile["p50"] >= tc.profile["p90"]).all()
    assert tc.ratio == pytest.approx(2.0)
    assert tc.eurs.idxmax() == 1004 and tc.eurs.idxmin() == 1000


def test_profile_stops_when_few_wells_remain():
    wells = synthetic_field()
    short = wells[~((wells["lease_no"] != 1004) & (wells["date"] >= "2015-06-01"))]  # only one well runs long
    tc = build_type_curve(short, min_months=6)
    assert tc.profile["n"].min() >= 3


def test_needs_three_wells():
    with pytest.raises(ValueError):
        build_type_curve(synthetic_field(scales=(1.0, 2.0)))


@pytest.mark.parametrize("path,product", [("spraberry_leases.csv", "oil"), ("barnett_gas_wells.csv", "gas")])
def test_bundled_example_fields(path, product):
    tc = build_type_curve(load_production(ROOT / "data" / path), product)
    assert tc.n_wells == 8
    assert 0 <= tc.fits["P50"].b <= 2
    assert tc.eurs.notna().all() and (tc.eurs > 0).all()


def test_pick_candidates_skips_non_producers():
    leases = parse_field_leases((FIXTURES / "pdq_field_oil.html").read_text())
    picked = pick_candidates(leases, "Oil", n=5)
    assert picked and all(w["oil"] > 0 for w in picked)
    assert not any(k in w["name"] for w in picked for k in ("SWD", "BRINE"))
    gas = pick_candidates(parse_field_leases((FIXTURES / "pdq_field_gas.html").read_text()), "Gas", n=5)
    assert gas and all(w["kind"] == "Gas" and w["gas"] > 0 for w in gas)
