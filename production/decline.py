"""Arps decline curves: fit to production history and forecast EUR.

Time t is in months and the decline rate Di is nominal per month.

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
from scipy.optimize import least_squares

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
    qi: float          # rate at the start of the fit, bbl/day
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


def prepare_history(df, start=None):
    """Monthly history for one lease to fit a decline to.

    start : first month of the fit window. Defaults to the month of peak oil
            rate. Set it later to fit only the decline after a workover or
            refrac, when the well's behaviour changed.

    Shut-in or unreported months (zero or missing oil) are dropped from the fit,
    because they reflect the well being off, not the reservoir declining. The
    time axis keeps calendar months, so the decline still accounts for elapsed time.
    """
    df = df.sort_values("date")
    producing = df[df["bopd"] > 0]
    if start is None:
        start = producing.loc[producing["bopd"].idxmax(), "date"]
    start = pd.Timestamp(start)
    hist = producing[producing["date"] >= start].copy()
    first = hist["date"].iloc[0]
    hist["t"] = (hist["date"].dt.year - first.year) * 12 + hist["date"].dt.month - first.month
    return hist


def fit_arps(hist, model="hyperbolic", b_max=2.0):
    """Fit one Arps model to history from prepare_history.

    Fits log(rate), so late low-rate months count as much as early high-rate
    ones. Uses a robust (soft L1) loss: months with partial downtime sit far
    below the trend and would otherwise drag the curve down.
    """
    t, q = hist["t"].to_numpy(float), hist["bopd"].to_numpy(float)
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
    below the economic limit q_limit (bbl/day) or max_years from the peak.

    d_term_annual is the terminal decline as an effective annual rate.
    Returns a DataFrame with date, t, bopd and monthly oil_bbl.
    """
    d_term = nominal_from_effective_annual(d_term_annual)
    t_last = int(hist["t"].iloc[-1])
    t = np.arange(t_last + 1, max_years * 12)
    q = fit.rate(t, d_term)
    keep = q >= q_limit
    t, q = t[keep], q[keep]

    dates = pd.DatetimeIndex([fit.t0 + pd.DateOffset(months=int(m)) for m in t])
    fc = pd.DataFrame({"date": dates, "t": t, "bopd": q})
    fc["oil_bbl"] = fc["bopd"] * fc["date"].dt.days_in_month
    return fc


def eur(df_lease, fc):
    """EUR (bbl): all actual oil produced on the lease plus the forecast."""
    return float(df_lease["oil_bbl"].sum(skipna=True) + fc["oil_bbl"].sum())
