"""Discounted cash flow (DCF) for an oil well.

Monthly cash flow:
    revenue   = oil × oil price + gas × gas price, after the price differential
    net rev.  = revenue × net revenue interest (NRI = working interest × (1 − royalty))
    taxes     = net revenue × (severance + ad valorem)
    LOE       = working interest × (fixed $/month + $/bbl × oil + $/Mcf × gas)
    net cash  = net revenue − taxes − LOE − capex

The well is shut in at the economic limit: the first month after which
operating cash flow stays negative.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass
class Assumptions:
    oil_price: float = 70.0          # $/bbl WTI, flat, for forecast months
    gas_price: float = 2.50          # $/MCF Henry Hub, flat
    oil_diff: float = -1.50          # $/bbl, realized price vs WTI (Midland is near WTI)
    gas_diff_pct: float = 0.70       # fraction of Henry Hub realized (casinghead gas sells at a discount)
    working_interest: float = 1.0
    royalty: float = 0.25            # typical Permian royalty
    severance_oil: float = 0.046     # Texas oil severance tax
    severance_gas: float = 0.075     # Texas gas severance tax
    ad_valorem: float = 0.02         # property tax, fraction of revenue
    loe_fixed: float = 8000.0        # $/month per well
    loe_var: float = 4.0             # $/bbl oil (water handling, trucking, chemicals)
    loe_var_gas: float = 0.0         # $/Mcf gas (compression, gathering); mainly for gas wells
    capex: float = 8_000_000.0       # drilling and completion, spent in month 0 (full-cycle only)
    discount_rate: float = 0.10      # annual; 10% gives the industry-standard PV10

    @property
    def nri(self):
        return self.working_interest * (1 - self.royalty)


def cash_flow(volumes, a, oil_prices=None, gas_prices=None, include_capex=False):
    """Monthly cash flow table.

    volumes    : DataFrame with date, oil_bbl, gas_mcf (gas optional)
    a          : Assumptions
    oil_prices : optional WTI $/bbl per month (aligned to volumes), e.g.
                 historical prices or a price deck; defaults to the flat a.oil_price
    gas_prices : the same for Henry Hub $/MCF; defaults to the flat a.gas_price
    include_capex : charge a.capex in the first month (full-cycle economics)
    """
    cf = volumes[["date"]].copy().reset_index(drop=True)
    oil = volumes["oil_bbl"].fillna(0).to_numpy(float)
    gas = volumes["gas_mcf"].fillna(0).to_numpy(float) if "gas_mcf" in volumes else np.zeros_like(oil)
    wti = np.asarray(oil_prices, float) if oil_prices is not None else np.full_like(oil, a.oil_price)
    hh = np.asarray(gas_prices, float) if gas_prices is not None else np.full_like(oil, a.gas_price)

    cf["oil_bbl"], cf["gas_mcf"] = oil, gas
    cf["oil_price"] = wti + a.oil_diff
    cf["gas_price"] = hh * a.gas_diff_pct
    oil_rev = a.nri * oil * cf["oil_price"]
    gas_rev = a.nri * gas * cf["gas_price"]
    cf["net_revenue"] = oil_rev + gas_rev
    cf["taxes"] = oil_rev * a.severance_oil + gas_rev * a.severance_gas + cf["net_revenue"] * a.ad_valorem
    cf["loe"] = a.working_interest * (a.loe_fixed + a.loe_var * oil + a.loe_var_gas * gas)
    cf["operating_cf"] = cf["net_revenue"] - cf["taxes"] - cf["loe"]

    # Economic limit: cut off after the last month with positive operating cash flow.
    # Capex is spent regardless, so full-cycle economics always keep month 0.
    positive = np.flatnonzero(cf["operating_cf"].to_numpy() > 0)
    n = positive[-1] + 1 if len(positive) else 0
    cf = cf.iloc[: max(n, 1 if include_capex else 0)]

    cf["capex"] = 0.0
    if include_capex and len(cf):
        cf.loc[cf.index[0], "capex"] = a.capex * a.working_interest
    cf["net_cf"] = cf["operating_cf"] - cf["capex"]
    months = np.arange(len(cf))
    cf["discount_factor"] = (1 + a.discount_rate) ** (-(months + 0.5) / 12)  # mid-month
    cf["pv"] = cf["net_cf"] * cf["discount_factor"]
    cf["cum_cf"] = cf["net_cf"].cumsum()
    return cf


def irr(cf):
    """Annual internal rate of return (the discount rate that makes NPV zero), or NaN."""
    flows = cf["net_cf"].to_numpy()
    if len(flows) == 0 or flows.min() >= 0 or flows.max() <= 0:
        return np.nan
    months = np.arange(len(flows)) + 0.5
    npv = lambda r: np.sum(flows * (1 + r) ** (-months / 12))
    # Find a bracket by stepping out from 0%: at extreme negative rates, tiny
    # late cash flows are magnified enormously and can flip the sign, so a
    # single wide bracket isn't reliable.
    up = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 25, 100]
    down = [0.0, -0.1, -0.2, -0.35, -0.5, -0.65, -0.8, -0.9]
    for grid in (up, down):
        vals = [npv(r) for r in grid]
        for (r0, v0), (r1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
            if v0 == 0:
                return r0
            if np.sign(v0) != np.sign(v1):
                return brentq(npv, min(r0, r1), max(r0, r1))
    return np.nan


@dataclass
class Summary:
    npv: float              # $ at the assumed discount rate
    undiscounted: float     # $ net cash, no discounting
    net_oil: float          # bbl produced until the economic limit (gross, 8/8ths)
    irr: float              # annual, NaN if not defined
    payout_months: float    # months until cumulative cash turns positive, NaN if never
    econ_limit: object      # last economic month


def summarize(cf):
    cum = cf["cum_cf"].to_numpy()
    paid = np.flatnonzero(cum >= 0)
    payout = float(paid[0] + 1) if len(paid) and cf["capex"].sum() > 0 else np.nan
    return Summary(
        npv=float(cf["pv"].sum()),
        undiscounted=float(cf["net_cf"].sum()),
        net_oil=float(cf["oil_bbl"].sum()),
        irr=irr(cf),
        payout_months=payout,
        econ_limit=cf["date"].iloc[-1] if len(cf) else None,
    )


BREAKEVEN_MAX = {"oil": 300.0, "gas": 50.0}  # $/bbl, $/MCF: above this, call it uneconomic


def breakeven_price(volumes, a, product="oil", oil_prices=None, gas_prices=None,
                    include_capex=True, lo=0.01, hi=None):
    """Flat price of the main product (WTI $/bbl for oil, Henry Hub $/MCF for gas)
    at which NPV is zero at the assumed discount rate. The other stream keeps
    its own prices (oil_prices / gas_prices, or the flat price in `a`).
    NaN if the well loses money even at `hi`."""
    hi = BREAKEVEN_MAX[product] if hi is None else hi

    def npv_at(p):
        if product == "oil":
            b, kw = replace(a, oil_price=p), dict(gas_prices=gas_prices)
        else:
            b, kw = replace(a, gas_price=p), dict(oil_prices=oil_prices)
        return cash_flow(volumes, b, include_capex=include_capex, **kw)["pv"].sum()

    if npv_at(hi) < 0:
        return np.nan
    if npv_at(lo) > 0:
        return lo
    return brentq(npv_at, lo, hi, xtol=0.01 if product == "oil" else 0.001)


def breakeven_oil_price(volumes, a, include_capex=True, lo=1.0, hi=300.0):
    """Flat WTI price ($/bbl) at which NPV is zero at the assumed discount rate."""
    return breakeven_price(volumes, a, "oil", include_capex=include_capex, lo=lo, hi=hi)


def price_deck(dates, years, prices):
    """Monthly prices from a yearly price deck.

    dates  : month start dates to price
    years  : deck years, ascending; prices : one price per year
    Months before the first deck year use the first price; months after the
    last deck year stay flat at the last price (the usual convention).
    """
    years = np.asarray(years, int)
    prices = np.asarray(prices, float)
    idx = np.searchsorted(years, pd.DatetimeIndex(dates).year, side="right") - 1
    return prices[np.clip(idx, 0, len(prices) - 1)]


def escalated_deck(start_year, n_years, first_price, escalation):
    """Yearly deck starting at first_price and growing escalation (fraction) per year."""
    years = np.arange(start_year, start_year + n_years)
    return years, first_price * (1 + escalation) ** np.arange(n_years)


def load_fred(path, name):
    """Monthly price series from a FRED CSV: columns date and `name`, missing values dropped."""
    p = pd.read_csv(path)
    p.columns = ["date", name]
    p["date"] = pd.to_datetime(p["date"])
    p[name] = pd.to_numeric(p[name], errors="coerce")
    return p.dropna()


def load_wti(path):
    """Monthly WTI spot price ($/bbl) from a FRED CSV (series MCOILWTICO)."""
    return load_fred(path, "wti")


def load_henry_hub(path):
    """Monthly Henry Hub spot gas price ($/MMBtu ≈ $/MCF) from FRED (series MHHNGSP)."""
    return load_fred(path, "hh")
