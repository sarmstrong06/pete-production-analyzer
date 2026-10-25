import numpy as np
import pandas as pd
import pytest

from production.appraisal import (
    forecast_volumes, full_volumes, priced_history, priced_wti, producing_gor,
    recovery_factor_check, secondary_ratio,
)


def lease():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "oil_bbl": [1000.0, 0.0, 800.0, np.nan],
        "gas_mcf": [2000.0, 500.0, 2400.0, 100.0],
    })


def gas_well():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="MS"),
        "gas_mcf": [30_000.0, 0.0, 20_000.0],
        "oil_bbl": [150.0, 10.0, 100.0],
    })


def forecast_df():
    return pd.DataFrame({"date": pd.date_range("2025-05-01", periods=2, freq="MS"),
                         "volume": [700.0, 600.0]})


def test_producing_gor_ignores_non_producing_months():
    # Only months with oil > 0 count: (2000 + 2400) / (1000 + 800)
    assert producing_gor(lease()) == pytest.approx(4400 / 1800)
    assert producing_gor(lease(), months=1) == pytest.approx(2400 / 800)


def test_condensate_yield_for_gas_wells():
    # bbl per Mcf over months with gas > 0: (150 + 100) / 50,000 = 5 bbl/MMcf
    assert secondary_ratio(gas_well(), "gas") == pytest.approx(0.005)
    assert secondary_ratio(gas_well(), "gas", months=None) == pytest.approx(0.005)


def test_forecast_volumes_oil_and_gas():
    vol = forecast_volumes(forecast_df(), 2.0)
    assert vol["oil_bbl"].tolist() == [700.0, 600.0]
    assert vol["gas_mcf"].tolist() == [1400.0, 1200.0]
    assert not vol["actual"].any()
    vol = forecast_volumes(forecast_df(), 0.005, product="gas")
    assert vol["gas_mcf"].tolist() == [700.0, 600.0]
    assert vol["oil_bbl"].tolist() == pytest.approx([3.5, 3.0])


def test_full_volumes_joins_history_and_forecast():
    vol = full_volumes(lease(), forecast_df())
    assert len(vol) == 6
    assert vol["actual"].tolist() == [True] * 4 + [False] * 2
    assert vol["oil_bbl"].iloc[4] == 700
    assert vol["gas_mcf"].iloc[4] == pytest.approx(700 * 4400 / 1800)


def test_full_volumes_gas_well():
    vol = full_volumes(gas_well(), forecast_df(), product="gas")
    assert vol["gas_mcf"].iloc[3] == 700
    assert vol["oil_bbl"].iloc[3] == pytest.approx(700 * 0.005)


def test_priced_wti_uses_history_then_flat():
    vol = full_volumes(lease(), forecast_df())
    wti = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=2, freq="MS"), "wti": [75.0, 72.0]})
    prices = priced_wti(vol, wti, flat_price=65.0)
    # Months 3-4 are actual but have no WTI yet, so they fall back to the flat price
    assert prices.tolist() == [75.0, 72.0, 65.0, 65.0, 65.0, 65.0]


def test_priced_history_with_a_deck():
    vol = full_volumes(lease(), forecast_df())
    hh = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=4, freq="MS"), "hh": [3.0, 3.1, 3.2, 3.3]})
    deck = np.array([9, 9, 9, 9, 4.0, 4.5])  # only the forecast months should use it
    assert priced_history(vol, hh, deck).tolist() == [3.0, 3.1, 3.2, 3.3, 4.0, 4.5]


@pytest.mark.parametrize("eur,verdict", [
    (80_000, "Typical"), (200_000, "Plausible"), (500_000, "Too high"),
    (5_000, "Very low"), (1_500_000, "Impossible"),
])
def test_recovery_factor_verdicts(eur, verdict):
    rf, text = recovery_factor_check(eur, 1_000_000)
    assert rf == pytest.approx(eur / 1_000_000)
    assert text.startswith(verdict)


def test_recovery_factor_uses_well_type_ranges():
    # 70% is typical for a conventional gas well, but impossible-looking for tight oil
    rf, text = recovery_factor_check(0.7, 1.0, typical=(0.5, 0.85), plausible=(0.3, 0.95),
                                     label="a conventional gas well")
    assert text == "Typical for a conventional gas well. Inputs are consistent."
    assert recovery_factor_check(0.7, 1.0)[1].startswith("Too high")


def test_recovery_factor_without_volume_in_place():
    rf, text = recovery_factor_check(1000, 0)
    assert np.isnan(rf) and "Nothing in place" in text
