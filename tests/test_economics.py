from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from production.economics import (
    Assumptions, breakeven_oil_price, breakeven_price, cash_flow, escalated_deck, irr,
    load_henry_hub, load_wti, price_deck, summarize,
)


def volumes(oil, gas=None):
    oil = np.asarray(oil, float)
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(oil), freq="MS"), "oil_bbl": oil})
    df["gas_mcf"] = 0.0 if gas is None else gas
    return df


def declining(n=240, q0=15_000, d=0.03):
    return volumes(q0 * np.exp(-d * np.arange(n)))


def test_nri():
    assert Assumptions(working_interest=0.5, royalty=0.2).nri == pytest.approx(0.4)


def test_one_month_hand_calculation():
    a = Assumptions()
    cf = cash_flow(volumes([1000]), a)
    row = cf.iloc[0]
    assert row["oil_price"] == pytest.approx(68.5)                 # 70 - 1.50
    assert row["net_revenue"] == pytest.approx(0.75 * 1000 * 68.5)  # 51,375
    assert row["taxes"] == pytest.approx(51_375 * (0.046 + 0.02))
    assert row["loe"] == pytest.approx(8000 + 4 * 1000)
    assert row["operating_cf"] == pytest.approx(51_375 - 3_390.75 - 12_000)
    assert row["discount_factor"] == pytest.approx(1.1 ** (-0.5 / 12))


def test_gas_revenue_and_tax():
    a = Assumptions()
    cf = cash_flow(volumes([1000], gas=[2000]), a)
    gas_rev = 0.75 * 2000 * 2.5 * 0.7
    assert cf["net_revenue"].iloc[0] == pytest.approx(0.75 * 1000 * 68.5 + gas_rev)
    expected_tax = 0.75 * 1000 * 68.5 * 0.046 + gas_rev * 0.075 + cf["net_revenue"].iloc[0] * 0.02
    assert cf["taxes"].iloc[0] == pytest.approx(expected_tax)


def test_economic_limit_cuts_uneconomic_tail():
    cf = cash_flow(volumes([5000, 4000, 100, 3000, 50, 20]), Assumptions())
    # Month index 2 is negative but production recovers; months 4-5 never do.
    assert len(cf) == 4


def test_capex_month_kept_even_if_never_economic():
    cf = cash_flow(volumes([10, 10]), Assumptions(), include_capex=True)
    assert len(cf) == 1
    assert cf["capex"].iloc[0] == pytest.approx(8_000_000)
    assert cf["net_cf"].iloc[0] < -8_000_000


def test_historical_prices_override_flat():
    cf = cash_flow(volumes([1000, 1000]), Assumptions(), oil_prices=[50, 90])
    assert cf["oil_price"].tolist() == pytest.approx([48.5, 88.5])


def test_npv_is_zero_at_the_irr():
    a = Assumptions()
    cf = cash_flow(declining(), a, include_capex=True)
    r = irr(cf)
    assert 0 < r < 5
    cf_at_irr = cash_flow(declining(), replace(a, discount_rate=r), include_capex=True)
    assert cf_at_irr["pv"].sum() == pytest.approx(0, abs=1.0)


def test_uneconomic_well_without_capex_has_no_cash_flow():
    """A worn-out well (PDP) whose every month loses money is shut in at once."""
    cf = cash_flow(volumes([50, 40]), Assumptions())
    assert cf.empty
    s = summarize(cf)
    assert s.npv == 0 and np.isnan(s.irr) and s.econ_limit is None


def test_irr_ignores_tiny_late_negative_months():
    """A small loss decades out must not hide the real IRR (it dominates at -99%)."""
    flows = np.r_[-5e6, np.full(60, 2e5), np.full(300, 500.0), -500.0, 50.0]
    cf = pd.DataFrame({"net_cf": flows})
    r = irr(cf)
    months = np.arange(len(flows)) + 0.5
    assert 0.1 < r < 1
    assert np.sum(flows * (1 + r) ** (-months / 12)) == pytest.approx(0, abs=1)


def test_negative_irr():
    flows = np.r_[-1e6, np.full(36, 2e4)]  # gets back 720k of 1M
    r = irr(pd.DataFrame({"net_cf": flows}))
    assert -0.5 < r < 0


def test_irr_undefined_without_sign_change():
    cf = cash_flow(declining(), Assumptions())  # no capex: all positive
    assert np.isnan(irr(cf))


def test_breakeven_gives_zero_npv():
    a = Assumptions()
    be = breakeven_oil_price(declining(), a)
    assert 10 < be < 300
    npv = lambda p: cash_flow(declining(), replace(a, oil_price=p), include_capex=True)["pv"].sum()
    assert npv(be - 0.02) < 0 < npv(be + 0.02)  # solver tolerance is 1 cent


def test_breakeven_nan_when_never_economic():
    assert np.isnan(breakeven_oil_price(volumes([100] * 12), Assumptions()))


def test_summarize_payout():
    s = summarize(cash_flow(declining(), Assumptions(), include_capex=True))
    assert s.payout_months > 1
    assert s.npv < s.undiscounted
    cf = cash_flow(declining(), Assumptions(), include_capex=True)
    assert cf["cum_cf"].iloc[int(s.payout_months) - 1] >= 0
    assert cf["cum_cf"].iloc[int(s.payout_months) - 2] < 0


def test_load_wti_skips_missing_values(tmp_path):
    p = tmp_path / "wti.csv"
    p.write_text("observation_date,MCOILWTICO\n2024-01-01,74.15\n2024-02-01,.\n2024-03-01,80.41\n")
    w = load_wti(p)
    assert list(w.columns) == ["date", "wti"]
    assert w["wti"].tolist() == pytest.approx([74.15, 80.41])


def test_load_henry_hub(tmp_path):
    p = tmp_path / "hh.csv"
    p.write_text("observation_date,MHHNGSP\n2024-01-01,3.18\n")
    assert list(load_henry_hub(p).columns) == ["date", "hh"]


# --- Gas wells ----------------------------------------------------------------

def gas_volumes(n=240, q0=60_000, d=0.03):
    gas = q0 * np.exp(-d * np.arange(n))
    return volumes(gas * 0.002, gas=gas)  # 2 bbl/MMcf condensate


def test_gas_variable_loe():
    a = Assumptions(loe_var_gas=0.5)
    cf = cash_flow(volumes([0], gas=[10_000]), a, include_capex=True)  # capex keeps the month
    assert cf["loe"].iloc[0] == pytest.approx(8000 + 0.5 * 10_000)


def test_gas_breakeven_gives_zero_npv():
    a = Assumptions(capex=3_000_000, gas_diff_pct=0.9, loe_var_gas=0.3)
    be = breakeven_price(gas_volumes(), a, product="gas")
    assert 0.5 < be < 50
    npv = lambda p: cash_flow(gas_volumes(), replace(a, gas_price=p), include_capex=True)["pv"].sum()
    assert npv(be - 0.002) < 0 < npv(be + 0.002)


def test_breakeven_oil_price_matches_general_version():
    a = Assumptions()
    assert breakeven_oil_price(declining(), a) == pytest.approx(breakeven_price(declining(), a), abs=0.02)


# --- Price decks --------------------------------------------------------------

def test_price_deck_holds_flat_after_last_year():
    dates = pd.to_datetime(["2025-06-01", "2026-01-01", "2026-12-01", "2027-03-01", "2035-01-01"])
    prices = price_deck(dates, [2026, 2027], [70.0, 75.0])
    assert prices.tolist() == [70.0, 70.0, 70.0, 75.0, 75.0]


def test_escalated_deck():
    years, prices = escalated_deck(2026, 3, 70.0, 0.10)
    assert years.tolist() == [2026, 2027, 2028]
    assert prices == pytest.approx([70.0, 77.0, 84.7])


def test_deck_prices_feed_cash_flow():
    vol = volumes([1000, 1000])
    cf = cash_flow(vol, Assumptions(), oil_prices=price_deck(vol["date"], [2026], [80.0]))
    assert cf["oil_price"].tolist() == pytest.approx([78.5, 78.5])
