"""Glue between the steps: combine history and forecast, and cross-check
log-based oil in place against decline-based EUR."""

import numpy as np
import pandas as pd

# Typical primary recovery factors for tight oil (horizontal, hydraulically
# fractured) are roughly 3-15% of oil in place. Outside 1-30% the inputs
# almost certainly don't describe the same rock volume.
RF_TYPICAL = (0.03, 0.15)
RF_PLAUSIBLE = (0.01, 0.30)


def producing_gor(lease_df, months=12):
    """Gas-oil ratio (MCF/bbl) over the last `months` producing months."""
    recent = lease_df.dropna(subset=["oil_bbl", "gas_mcf"])
    recent = recent[recent["oil_bbl"] > 0].tail(months)
    return float(recent["gas_mcf"].sum() / recent["oil_bbl"].sum()) if len(recent) else 0.0


def full_volumes(lease_df, fc):
    """Actual monthly volumes followed by the forecast, in one table.

    Forecast gas = forecast oil × recent gas-oil ratio.
    """
    hist = lease_df[["date", "oil_bbl", "gas_mcf"]].copy()
    hist["actual"] = True
    future = fc[["date", "oil_bbl"]].copy()
    future["gas_mcf"] = future["oil_bbl"] * producing_gor(lease_df)
    future["actual"] = False
    return pd.concat([hist, future], ignore_index=True)


def priced_wti(volumes, wti, flat_price):
    """WTI per month: historical price for past months, flat price for the forecast."""
    hist = volumes[["date"]].merge(wti, on="date", how="left")["wti"]
    return np.where(volumes["actual"] & hist.notna(), hist, flat_price)


def recovery_factor_check(eur_bbl, ooip_stb):
    """Recovery factor (EUR / OOIP) and a plain-language verdict."""
    if not ooip_stb or ooip_stb <= 0:
        return np.nan, "No oil in place: check the log inputs."
    rf = eur_bbl / ooip_stb
    lo, hi = RF_PLAUSIBLE
    tlo, thi = RF_TYPICAL
    if rf > 1:
        verdict = ("Impossible: the well would produce more oil than is in place. "
                   "The drainage area or net pay is too small for this well, or the log "
                   "doesn't describe the rock this well produces from.")
    elif rf > hi:
        verdict = ("Too high for tight oil. The drainage area or net pay is probably "
                   "understated, or the decline forecast is too optimistic.")
    elif rf < lo:
        verdict = ("Very low. The drainage area or net pay may be overstated, or the "
                   "forecast too pessimistic.")
    elif tlo <= rf <= thi:
        verdict = "Typical for a horizontal tight-oil well. Inputs are consistent."
    else:
        verdict = "Plausible but outside the typical 3-15% range; worth a second look."
    return rf, verdict
