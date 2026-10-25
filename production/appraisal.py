"""Glue between the steps: combine history and forecast, and cross-check
log-based volumes in place against decline-based EUR.

A well's main product is oil or gas. The decline is fitted to the main
product; the other stream (casinghead gas for an oil well, condensate for a
gas well) is forecast as a fixed ratio to it.
"""

import numpy as np
import pandas as pd

# Columns for each main product: daily rate fitted, monthly volume, other stream
PRODUCTS = {
    "oil": dict(rate="bopd", volume="oil_bbl", other="gas_mcf"),
    "gas": dict(rate="gas_mcfd", volume="gas_mcf", other="oil_bbl"),
}

# Default recovery factor ranges (typical, plausible), tight oil
RF_TYPICAL = (0.03, 0.15)
RF_PLAUSIBLE = (0.01, 0.30)


def secondary_ratio(lease_df, product="oil", months=12):
    """Other stream per unit of main product over the last `months` producing
    months (all of them if months is None): gas-oil ratio in Mcf/bbl for an oil
    well, condensate yield in bbl/Mcf for a gas well."""
    p = PRODUCTS[product]
    recent = lease_df.dropna(subset=[p["volume"], p["other"]])
    recent = recent[recent[p["volume"]] > 0]
    if months is not None:
        recent = recent.tail(months)
    total = recent[p["volume"]].sum()
    return float(recent[p["other"]].sum() / total) if total else 0.0


def producing_gor(lease_df, months=12):
    """Gas-oil ratio (Mcf/bbl) of an oil lease over the last `months` producing months."""
    return secondary_ratio(lease_df, "oil", months)


def _split(volume, ratio, product):
    """Oil and gas columns from the main product's volume and the other stream's ratio."""
    other = volume * ratio
    return (volume, other) if product == "oil" else (other, volume)


def full_volumes(lease_df, fc, product="oil", ratio=None):
    """Actual monthly volumes followed by the forecast, in one table.

    The forecast's other stream = forecast main product × ratio (default: the
    ratio over the last 12 producing months).
    """
    if ratio is None:
        ratio = secondary_ratio(lease_df, product)
    hist = lease_df[["date", "oil_bbl", "gas_mcf"]].copy()
    hist["actual"] = True
    future = fc[["date"]].copy()
    future["oil_bbl"], future["gas_mcf"] = _split(fc["volume"].to_numpy(), ratio, product)
    future["actual"] = False
    return pd.concat([hist, future], ignore_index=True)


def forecast_volumes(fc, ratio, product="oil"):
    """Volumes for a well with no history: the forecast main product, and the
    other stream = main product × ratio."""
    vol = fc[["date"]].copy()
    vol["oil_bbl"], vol["gas_mcf"] = _split(fc["volume"].to_numpy(), ratio, product)
    vol["actual"] = False
    return vol


def priced_history(volumes, history, forecast_price):
    """Price per month: the historical price for past months, `forecast_price`
    (one number, or one value per row of `volumes`) for forecast months.

    history : DataFrame with date and one price column (e.g. load_wti)
    """
    col = [c for c in history.columns if c != "date"][0]
    past = volumes[["date"]].merge(history, on="date", how="left")[col].to_numpy()
    future = np.broadcast_to(np.asarray(forecast_price, float), past.shape)
    return np.where(volumes["actual"].to_numpy() & ~np.isnan(past), past, future)


def priced_wti(volumes, wti, flat_price):
    """WTI per month: historical price for past months, flat price for the forecast."""
    return priced_history(volumes, wti, flat_price)


def recovery_factor_check(eur, in_place, typical=RF_TYPICAL, plausible=RF_PLAUSIBLE,
                          label="a horizontal tight-oil well"):
    """Recovery factor (EUR / volume in place) and a plain-language verdict.

    typical, plausible : (low, high) recovery factor ranges for the well type
    label              : how the well type reads in the verdict
    """
    if not in_place or in_place <= 0:
        return np.nan, "Nothing in place: check the log inputs."
    rf = eur / in_place
    lo, hi = plausible
    tlo, thi = typical
    if rf > 1:
        verdict = ("Impossible: the well would produce more than is in place. "
                   "The drainage area or net pay is too small for this well, or the log "
                   "doesn't describe the rock this well produces from.")
    elif rf > hi:
        verdict = (f"Too high for {label}. The drainage area or net pay is probably "
                   "understated, or the decline forecast is too optimistic.")
    elif rf < lo:
        verdict = ("Very low. The drainage area or net pay may be overstated, or the "
                   "forecast too pessimistic.")
    elif tlo <= rf <= thi:
        verdict = f"Typical for {label}. Inputs are consistent."
    else:
        verdict = (f"Plausible but outside the typical {tlo:.0%}-{thi:.0%} range for {label}; "
                   "worth a second look.")
    return rf, verdict
