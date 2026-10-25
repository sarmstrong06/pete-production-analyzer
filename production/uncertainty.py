"""Monte Carlo uncertainty: run the appraisal thousands of times with uncertain inputs.

Each uncertain input gets a triangular distribution (low, most likely, high).
The results are reported the industry way:
    P90 = low case    (90% chance of doing at least this well)
    P50 = median
    P10 = high case   (10% chance of doing better)

For speed, every simulation is computed at once as a matrix (simulations x
months) with the same cash flow rules as economics.cash_flow, which a test
checks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from production.decline import DeclineFit, nominal_from_effective_annual, nominal_from_effective_initial


@dataclass
class Range:
    """A triangular distribution. low == high means a fixed value."""
    low: float
    mode: float
    high: float

    def sample(self, rng, n):
        lo, md, hi = sorted([self.low, self.high])[0], self.mode, max(self.low, self.high)
        md = min(max(md, lo), hi)
        if hi - lo < 1e-12:
            return np.full(n, md, dtype=float)
        return rng.triangular(lo, md, hi, n)


def percentiles(x):
    """P90 (low), P50, P10 (high) of a sample, industry convention."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return {"P90": float(np.percentile(x, 10)), "P50": float(np.percentile(x, 50)),
            "P10": float(np.percentile(x, 90))} if len(x) else {"P90": np.nan, "P50": np.nan, "P10": np.nan}


def npv_paths(oil, gas, oil_price, gas_price, a, loe_fixed=None, capex=None, include_capex=True):
    """NPV of many production paths at once, with the rules of economics.cash_flow.

    oil, gas             : (sims, months) volumes per month
    oil_price, gas_price : WTI and Henry Hub, broadcastable to (sims, months)
    a                    : economics.Assumptions (fiscal terms, discount rate, defaults)
    loe_fixed, capex     : optional per-simulation arrays (sims,) overriding a
    Returns a dict: npv (sims,) and keep, the (sims, months) mask of months
    before the economic limit.
    """
    oil, gas = np.atleast_2d(np.nan_to_num(oil)), np.atleast_2d(np.nan_to_num(gas))
    s, m = oil.shape
    lf = np.full((s, 1), a.loe_fixed) if loe_fixed is None else np.asarray(loe_fixed, float).reshape(s, 1)
    cx = np.full(s, a.capex) if capex is None else np.asarray(capex, float).reshape(s)
    oil_p = np.broadcast_to(oil_price, (s, m)) + a.oil_diff
    gas_p = np.broadcast_to(gas_price, (s, m)) * a.gas_diff_pct
    oil_rev, gas_rev = a.nri * oil * oil_p, a.nri * gas * gas_p
    net = oil_rev + gas_rev
    taxes = oil_rev * a.severance_oil + gas_rev * a.severance_gas + net * a.ad_valorem
    loe = a.working_interest * (lf + a.loe_var * oil + a.loe_var_gas * gas)
    op = net - taxes - loe

    # Economic limit: keep months up to the last one with positive operating cash flow
    positive = op > 0
    last = np.where(positive.any(axis=1), m - 1 - np.argmax(positive[:, ::-1], axis=1), -1)
    keep = np.arange(m)[None, :] <= last[:, None]
    cf = np.where(keep, op, 0.0)
    if include_capex:  # capex is spent regardless, so month 0 is always kept
        keep[:, 0] = True
        cf[:, 0] = op[:, 0] - cx * a.working_interest
    disc = (1 + a.discount_rate) ** (-(np.arange(m) + 0.5) / 12)
    return {"npv": (cf * disc).sum(axis=1), "keep": keep}


def unit_volumes(start, b, d_eff_initial, d_term_annual, max_years=50):
    """Monthly volume of a decline with qi = 1/day, and its daily rate, for scaling."""
    di = nominal_from_effective_initial(d_eff_initial, b)
    fit = DeclineFit("unit", 1.0, di, b, pd.Timestamp(start), np.nan)
    t = np.arange(max_years * 12)
    rate = fit.rate(t, nominal_from_effective_annual(d_term_annual))
    days = pd.DatetimeIndex([pd.Timestamp(start) + pd.DateOffset(months=int(k)) for k in t]).days_in_month
    return rate, rate * days.to_numpy(float)


def scaled_forecasts(eur, rate, unit_vol, q_limit):
    """Monthly volumes (sims, months) for target EURs with one decline shape.

    The initial rate qi is chosen so the volume produced above q_limit equals
    each target EUR (the same rule as decline.type_curve), via interpolation
    on a grid of qi values.
    """
    eur = np.asarray(eur, float)
    grid = np.geomspace(q_limit * 1.0001, 1e6, 600)
    above = rate[None, :] * grid[:, None] >= q_limit
    vol = (unit_vol[None, :] * grid[:, None] * above).sum(axis=1)
    qi = np.interp(eur, vol, grid, left=0.0, right=grid[-1])
    out = qi[:, None] * unit_vol[None, :]
    out[qi[:, None] * rate[None, :] < q_limit] = 0.0
    return out


def tornado(base_npv, run, ranges):
    """NPV with each input at its low and high value, the others at most likely.

    run(name, value) -> NPV. Returns a DataFrame sorted by swing, largest first.
    """
    rows = []
    for name, r in ranges.items():
        lo, hi = run(name, r.low), run(name, r.high)
        rows.append({"input": name, "low": lo, "high": hi, "swing": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
