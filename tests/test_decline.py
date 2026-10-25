import numpy as np
import pandas as pd
import pytest

from production.decline import (
    DeclineFit, arps_rate, eur, fit_arps, forecast, nominal_from_effective_annual,
    nominal_from_effective_initial, prepare_history, type_curve,
)


def synthetic_lease(qi=500.0, di=0.15, b=0.9, months=60, ramp=2):
    """Monthly lease history: a short ramp-up, then an exact Arps decline from the peak."""
    dates = pd.date_range("2020-01-01", periods=months + ramp, freq="MS")
    t = np.arange(len(dates)) - ramp
    bopd = np.where(t < 0, qi * 0.5, arps_rate(np.maximum(t, 0), qi, di, b))
    df = pd.DataFrame({"date": dates, "bopd": bopd})
    df["oil_bbl"] = df["bopd"] * df["date"].dt.days_in_month
    return df


# --- Arps equations -----------------------------------------------------------

def test_arps_special_cases():
    t = np.array([0, 6, 24])
    assert arps_rate(t, 100, 0.1, 0) == pytest.approx(100 * np.exp(-0.1 * t))
    assert arps_rate(t, 100, 0.1, 1) == pytest.approx(100 / (1 + 0.1 * t))
    assert arps_rate(0, 100, 0.1, 0.7) == pytest.approx(100)


def test_nominal_from_effective_annual():
    d = nominal_from_effective_annual(0.08)
    assert np.exp(-12 * d) == pytest.approx(0.92)


@pytest.mark.parametrize("b", [0.0, 0.5, 1.0, 1.5])
def test_effective_initial_round_trip(b):
    """Converting a first-year decline to nominal Di and back gives the same number."""
    di = nominal_from_effective_initial(0.6, b)
    fit = DeclineFit("x", 100, di, b, pd.Timestamp("2020-01-01"), 0)
    assert fit.di_annual_eff == pytest.approx(0.6)
    assert arps_rate(12, 100, di, b) == pytest.approx(40)


def test_terminal_decline_switch():
    fit = DeclineFit("hyperbolic", 500, 0.2, 1.3, pd.Timestamp("2020-01-01"), 0)
    d_term = nominal_from_effective_annual(0.08)
    t_sw = (fit.di / d_term - 1) / (fit.b * fit.di)
    t = np.array([t_sw - 1e-6, t_sw + 1e-6])
    q = fit.rate(t, d_term)
    assert q[0] == pytest.approx(q[1], rel=1e-6)             # continuous at the switch
    late = fit.rate(np.array([400.0, 401.0]), d_term)
    assert late[1] / late[0] == pytest.approx(np.exp(-d_term))  # exponential afterwards
    assert fit.rate(400.0, d_term) < fit.rate(400.0)          # below plain hyperbolic


# --- History and fitting ------------------------------------------------------

def test_prepare_history_starts_at_peak_and_skips_shut_ins():
    df = synthetic_lease(months=24)
    df.loc[10, "bopd"] = 0          # shut-in month
    df.loc[11, "bopd"] = np.nan     # unreported month
    hist = prepare_history(df)
    assert hist["date"].iloc[0] == df["date"].iloc[2]   # peak, after the 2-month ramp
    assert hist["t"].iloc[0] == 0
    assert len(hist) == 22
    assert hist.loc[12, "t"] == 10  # calendar months still counted across the gap


def test_prepare_history_custom_start():
    df = synthetic_lease(months=24)
    hist = prepare_history(df, start="2021-01-01")
    assert hist["date"].iloc[0] == pd.Timestamp("2021-01-01") and hist["t"].iloc[0] == 0


def test_hyperbolic_fit_recovers_parameters():
    fit = fit_arps(prepare_history(synthetic_lease(qi=500, di=0.15, b=0.9)), "hyperbolic")
    assert fit.qi == pytest.approx(500, rel=0.01)
    assert fit.di == pytest.approx(0.15, rel=0.03)
    assert fit.b == pytest.approx(0.9, abs=0.03)
    assert fit.rmse_log < 1e-3


@pytest.mark.parametrize("model,b", [("exponential", 0.0), ("harmonic", 1.0)])
def test_fixed_b_fits(model, b):
    fit = fit_arps(prepare_history(synthetic_lease(qi=300, di=0.05, b=b)), model)
    assert fit.b == b
    assert fit.di == pytest.approx(0.05, rel=0.02)


def test_fit_is_robust_to_partial_downtime():
    df = synthetic_lease(qi=500, di=0.15, b=0.9)
    df.loc[[15, 27, 40], "bopd"] *= 0.3   # three months mostly shut in
    fit = fit_arps(prepare_history(df), "hyperbolic")
    assert fit.b == pytest.approx(0.9, abs=0.1)
    assert fit.qi == pytest.approx(500, rel=0.03)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        fit_arps(prepare_history(synthetic_lease()), "linear")


# --- Forecast and EUR ---------------------------------------------------------

def test_forecast_continues_after_history_and_stops_at_limit():
    df = synthetic_lease()
    hist = prepare_history(df)
    fit = fit_arps(hist, "hyperbolic")
    fc = forecast(fit, hist, q_limit=5)
    assert fc["t"].iloc[0] == hist["t"].iloc[-1] + 1
    assert fc["date"].iloc[0] == hist["date"].iloc[-1] + pd.DateOffset(months=1)
    assert fc["q"].min() >= 5
    assert fc["q"].is_monotonic_decreasing
    assert fc["volume"].iloc[0] == pytest.approx(fc["q"].iloc[0] * fc["date"].iloc[0].days_in_month)
    assert eur(df, fc) == pytest.approx(df["oil_bbl"].sum() + fc["volume"].sum())


def test_gas_rate_column():
    """A gas well fits its Mcf/day rate the same way an oil well fits bbl/day."""
    df = synthetic_lease(qi=3000, di=0.12, b=1.2).rename(columns={"bopd": "gas_mcfd", "oil_bbl": "gas_mcf"})
    hist = prepare_history(df, rate="gas_mcfd")
    fit = fit_arps(hist, "hyperbolic")
    assert fit.qi == pytest.approx(3000, rel=0.01) and fit.b == pytest.approx(1.2, abs=0.03)
    fc = forecast(fit, hist, q_limit=50)
    assert eur(df, fc, "gas_mcf") == pytest.approx(df["gas_mcf"].sum() + fc["volume"].sum())


def test_type_curve_hits_target_eur():
    fit, fc = type_curve(300_000, "2026-01-01", b=1.1, d_eff_initial=0.7)
    assert fc["volume"].sum() == pytest.approx(300_000, rel=1e-3)
    assert fit.di_annual_eff == pytest.approx(0.7)
    assert fc["date"].iloc[0] == pd.Timestamp("2026-01-01")


def test_type_curve_edge_cases():
    _, fc = type_curve(0, "2026-01-01", b=1.0, d_eff_initial=0.6)
    assert fc.empty
    with pytest.raises(ValueError):
        type_curve(1e12, "2026-01-01", b=1.0, d_eff_initial=0.6)
