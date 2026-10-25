from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from production.economics import (
    Assumptions, breakeven_oil_price, cash_flow, irr, load_wti, summarize,
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
