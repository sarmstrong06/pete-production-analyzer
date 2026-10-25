"""Arps decline curves: fit to production history and forecast EUR.

Time t is in months and the decline rate Di is nominal per month. Rates are
per day in the well's main product: bbl/day for an oil well, Mcf/day for a gas
well. History tables carry the rate being fitted as column "q"; forecasts
return "q" and the monthly "volume" (bbl or Mcf).

    Exponential (b = 0) : q = qi * exp(-Di t)
    Hyperbolic (0<b<1+) : q = qi / (1 + b Di t)^(1/b)
    Harmonic   (b = 1)  : q = qi / (1 + Di t)

Shale and tight wells often fit with b > 1, which on its own forecasts an
infinite EUR. The standard fix (modified hyperbolic) is used here: once the
instantaneous decline rate falls to a terminal value, the forecast switches to
exponential decline at that rate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

MODELS = ("exponential", "hyperbolic", "harmonic")


def arps_rate(t, qi, di, b):
    """Arps rate at time t (months). b near 0 is treated as exponential."""
    t = np.asarray(t, dtype=float)
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / (1 + b * di * t) ** (1 / b)


def nominal_from_effective_annual(d_eff):
    """Convert an effective annual decline (e.g. 0.06 = 6%/yr) to nominal per month."""
    return -np.log(1 - d_eff) / 12


@dataclass
class DeclineFit:
    model: str
    qi: float          # rate at the start of the fit, per day (bbl or Mcf)
    di: float          # initial nominal decline, per month
    b: float           # Arps b-factor
    t0: object         # date of the first month in the fit (the peak)
    rmse_log: float    # fit error, in natural-log units of rate

    @property
    def di_annual_eff(self):
        """Initial decline as an effective annual percentage, the usual way it's quoted."""
        return 1 - np.exp(-self.di * 12) if self.b < 1e-6 else 1 - (1 + self.b * self.di * 12) ** (-1 / self.b)

    def rate(self, t, d_term=None):
        """Forecast rate at months t since t0, switching to exponential at the
        terminal decline d_term (nominal per month) if given."""
        t = np.asarray(t, dtype=float)
        q = arps_rate(t, self.qi, self.di, self.b)
        if d_term is None or self.b < 1e-6 or d_term >= self.di:
            return q
        # Instantaneous decline D(t) = di / (1 + b di t) reaches d_term at t_sw
        t_sw = (self.di / d_term - 1) / (self.b * self.di)
        q_sw = arps_rate(t_sw, self.qi, self.di, self.b)
        return np.where(t < t_sw, q, q_sw * np.exp(-d_term * (t - t_sw)))


def prepare_history(df, start=None, rate="bopd"):
    """Monthly history for one lease to fit a decline to.

    start : first month of the fit window. Defaults to the month of peak
            rate. Set it later to fit only the decline after a workover or
            refrac, when the well's behaviour changed.
    rate  : column with the daily rate to fit: "bopd" for oil, "gas_mcfd" for gas.
            It is copied to column "q".

    Shut-in or unreported months (zero or missing rate) are dropped from the fit,
    because they reflect the well being off, not the reservoir declining. The
    time axis keeps calendar months, so the decline still accounts for elapsed time.
    """
    df = df.sort_values("date")
    producing = df[df[rate] > 0]
    if start is None:
        start = producing.loc[producing[rate].idxmax(), "date"]
    start = pd.Timestamp(start)
    hist = producing[producing["date"] >= start].copy()
    hist["q"] = hist[rate]
    first = hist["date"].iloc[0]
    hist["t"] = (hist["date"].dt.year - first.year) * 12 + hist["date"].dt.month - first.month
    return hist


def fit_arps(hist, model="hyperbolic", b_max=2.0):
    """Fit one Arps model to history from prepare_history.

    Fits log(rate), so late low-rate months count as much as early high-rate
    ones. Uses a robust (soft L1) loss: months with partial downtime sit far
    below the trend and would otherwise drag the curve down.
    """
    t, q = hist["t"].to_numpy(float), hist["q"].to_numpy(float)
    t = t - t[0]  # time from the first month in the fit window
    q0 = q[0]

    if model == "exponential":
        b_lo, b_hi, b0 = 0.0, 1e-9, 0.0
    elif model == "harmonic":
        b_lo, b_hi, b0 = 1.0, 1.0 + 1e-9, 1.0
    elif model == "hyperbolic":
        b_lo, b_hi, b0 = 0.01, b_max, 1.0
    else:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")

    def resid(p):
        return np.log(arps_rate(t, p[0], p[1], p[2])) - np.log(q)

    res = least_squares(resid, x0=[q0, 0.1, b0], loss="soft_l1", f_scale=0.3,
                        bounds=([q0 * 0.1, 1e-5, b_lo], [q0 * 3, 5, b_hi]))
    qi, di, b = res.x
    if model != "hyperbolic":
        b = b_lo
    rmse = float(np.sqrt(np.mean(resid([qi, di, b]) ** 2)))
    return DeclineFit(model, qi, di, b, hist["date"].iloc[0], rmse)


def forecast(fit, hist, q_limit=5.0, d_term_annual=0.08, max_years=50):
    """Monthly forecast from the month after the last actual until the rate drops
    below the cutoff q_limit (per day) or max_years from the peak.

    d_term_annual is the terminal decline as an effective annual rate.
    hist=None forecasts from month 0 (a well with no production yet).
    Returns a DataFrame with date, t, daily rate q and monthly volume.
    """
    d_term = nominal_from_effective_annual(d_term_annual)
    t_last = int(hist["t"].iloc[-1]) if hist is not None else -1
    t = np.arange(t_last + 1, max_years * 12)
    q = fit.rate(t, d_term)
    keep = q >= q_limit
    t, q = t[keep], q[keep]

    dates = pd.DatetimeIndex([fit.t0 + pd.DateOffset(months=int(m)) for m in t])
    fc = pd.DataFrame({"date": dates, "t": t, "q": q})
    fc["volume"] = fc["q"] * fc["date"].dt.days_in_month
    return fc


def eur(df_lease, fc, volume="oil_bbl"):
    """EUR: all actual production on the lease (column `volume`, e.g. oil_bbl
    or gas_mcf) plus the forecast."""
    return float(df_lease[volume].sum(skipna=True) + fc["volume"].sum())


def nominal_from_effective_initial(d_eff, b):
    """Nominal monthly Di from an effective annual initial decline (first-year
    drop, e.g. 0.6 = rate falls 60% in year one) for a given b-factor."""
    if b < 1e-6:
        return nominal_from_effective_annual(d_eff)
    return ((1 - d_eff) ** (-b) - 1) / (12 * b)


def type_curve(target_eur, start, b, d_eff_initial, d_term_annual=0.08, q_limit=5.0, max_years=50):
    """Production forecast for an undrilled well, sized so it recovers target_eur.

    With no production history, the shape of the decline (b, initial decline)
    comes from analog wells, and the initial rate qi is solved so the volume
    produced before the rate drops below q_limit equals target_eur (bbl or Mcf,
    matching q_limit's per-day unit).
    Returns (DeclineFit, forecast DataFrame); the forecast is empty if the
    target is too small to produce above q_limit at all.
    """
    di = nominal_from_effective_initial(d_eff_initial, b)
    start = pd.Timestamp(start)

    def make(qi):
        fit = DeclineFit("type curve", qi, di, b, start, np.nan)
        return fit, forecast(fit, None, q_limit=q_limit, d_term_annual=d_term_annual, max_years=max_years)

    def shortfall(qi):
        return make(qi)[1]["volume"].sum() - target_eur

    lo, hi = q_limit * 1.0001, 1e6
    if target_eur <= 0 or shortfall(lo) > 0:
        return make(lo)[0], make(lo)[1].iloc[:0]
    if shortfall(hi) < 0:
        raise ValueError(f"An EUR of {target_eur:,.0f} needs an initial rate above {hi:,.0f} per day.")
    return make(brentq(shortfall, lo, hi, xtol=0.01))
