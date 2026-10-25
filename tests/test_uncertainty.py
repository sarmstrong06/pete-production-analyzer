from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from production.decline import type_curve
from production.economics import Assumptions, cash_flow
from production.uncertainty import (Range, npv_paths, percentiles, scaled_forecasts, tornado,
                                    unit_volumes)


def test_range_sampling():
    rng = np.random.default_rng(0)
    x = Range(10, 20, 40).sample(rng, 20_000)
    assert x.min() >= 10 and x.max() <= 40
    assert x.mean() == pytest.approx((10 + 20 + 40) / 3, rel=0.02)   # triangular mean
    assert (Range(5, 5, 5).sample(rng, 3) == 5).all()                 # fixed value
    assert Range(40, 20, 10).sample(rng, 100).min() >= 10             # low/high in either order


def test_percentiles_follow_industry_convention():
    p = percentiles(np.arange(1, 101))
    assert p["P90"] < p["P50"] < p["P10"]       # P90 is the low case
    assert p["P50"] == pytest.approx(50.5)


@pytest.mark.parametrize("include_capex", [True, False])
@pytest.mark.parametrize("q0", [15_000, 300, 20])   # economic, partly economic, never economic
def test_vectorized_npv_matches_cash_flow(include_capex, q0):
    a = Assumptions(loe_var_gas=0.2)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="MS")
    oil = q0 * np.exp(-0.03 * np.arange(n))
    gas = oil * 2.0
    wti = np.linspace(60, 80, n)
    hh = np.linspace(2.5, 3.5, n)
    vol = pd.DataFrame({"date": dates, "oil_bbl": oil, "gas_mcf": gas})
    expected = cash_flow(vol, a, oil_prices=wti, gas_prices=hh, include_capex=include_capex)["pv"].sum()
    got = npv_paths(oil[None, :], gas[None, :], wti, hh, a, include_capex=include_capex)["npv"][0]
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_npv_paths_per_simulation_overrides():
    a = Assumptions()
    oil = np.tile(10_000 * np.exp(-0.03 * np.arange(60)), (2, 1))
    r = npv_paths(oil, oil * 0, 70.0, 3.0, a, capex=[8e6, 4e6])["npv"]
    assert r[1] - r[0] == pytest.approx(4e6 * 1.1 ** (-0.5 / 12))  # capex in month 0, discounted mid-month


def test_scaled_forecasts_match_type_curve():
    start, b, d1, dt, ql = "2026-01-01", 1.1, 0.7, 0.08, 5.0
    rate, uvol = unit_volumes(start, b, d1, dt)
    vols = scaled_forecasts([100_000, 300_000], rate, uvol, ql)
    assert vols.sum(axis=1) == pytest.approx([100_000, 300_000], rel=2e-3)
    _, fc = type_curve(300_000, start, b, d1, d_term_annual=dt, q_limit=ql)
    assert vols[1, :len(fc)] == pytest.approx(fc["volume"].to_numpy(), rel=5e-3)


def test_tornado_orders_by_swing():
    ranges = {"small": Range(9, 10, 11), "big": Range(0, 10, 30)}
    t = tornado(0, lambda name, v: v * (1 if name == "small" else 3), ranges)
    assert t["input"].tolist() == ["big", "small"]
    assert t.loc[0, "swing"] == pytest.approx(90)
