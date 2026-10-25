import numpy as np
import pandas as pd
import pytest

from production.appraisal import (
    forecast_volumes, full_volumes, priced_wti, producing_gor, recovery_factor_check,
)


def lease():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "oil_bbl": [1000.0, 0.0, 800.0, np.nan],
        "gas_mcf": [2000.0, 500.0, 2400.0, 100.0],
    })


def forecast_df():
    return pd.DataFrame({"date": pd.date_range("2025-05-01", periods=2, freq="MS"),
                         "oil_bbl": [700.0, 600.0]})


def test_producing_gor_ignores_non_producing_months():
    # Only months with oil > 0 count: (2000 + 2400) / (1000 + 800)
    assert producing_gor(lease()) == pytest.approx(4400 / 1800)
    assert producing_gor(lease(), months=1) == pytest.approx(2400 / 800)


def test_forecast_volumes():
    vol = forecast_volumes(forecast_df(), gor=2.0)
    assert vol["gas_mcf"].tolist() == [1400.0, 1200.0]
    assert not vol["actual"].any()


def test_full_volumes_joins_history_and_forecast():
    vol = full_volumes(lease(), forecast_df())
    assert len(vol) == 6
    assert vol["actual"].tolist() == [True] * 4 + [False] * 2
    assert vol["gas_mcf"].iloc[4] == pytest.approx(700 * 4400 / 1800)


def test_priced_wti_uses_history_then_flat():
    vol = full_volumes(lease(), forecast_df())
    wti = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=2, freq="MS"), "wti": [75.0, 72.0]})
    prices = priced_wti(vol, wti, flat_price=65.0)
    # Months 3-4 are actual but have no WTI yet, so they fall back to the flat price
    assert prices.tolist() == [75.0, 72.0, 65.0, 65.0, 65.0, 65.0]


@pytest.mark.parametrize("eur,verdict", [
    (80_000, "Typical"), (200_000, "Plausible"), (500_000, "Too high"),
    (5_000, "Very low"), (1_500_000, "Impossible"),
])
def test_recovery_factor_verdicts(eur, verdict):
    rf, text = recovery_factor_check(eur, 1_000_000)
    assert rf == pytest.approx(eur / 1_000_000)
    assert text.startswith(verdict)


def test_recovery_factor_without_oil_in_place():
    rf, text = recovery_factor_check(1000, 0)
    assert np.isnan(rf) and "No oil in place" in text
