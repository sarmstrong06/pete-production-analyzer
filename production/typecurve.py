"""Field type curves: the typical decline of many wells in one field.

Each well's production is lined up by months since its peak, so wells drilled
in different years start together at month 0. At every month the spread of
rates across the wells gives the P10 (high), P50 (median) and P90 (low)
profiles. Fitting an Arps curve to the P50 profile gives the field's typical
decline shape, which a new well, sized by its log, can follow.

P10/P50/P90 follow the industry convention: P10 is the high case (10% of wells
do better), P90 the low case (90% of wells do better).
"""

import random
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from production.appraisal import PRODUCTS, secondary_ratio
from production.decline import eur, fit_arps, forecast, prepare_history
from production.rrc import PDQSession, clean_production, fetch_lease

NOT_PRODUCERS = re.compile(r"\b(SWD|BRINE|DISPOSAL|INJ|INJECTION|WSW|WATER SUPPLY)\b", re.I)


@dataclass
class TypeCurve:
    profile: pd.DataFrame     # t, p10, p50, p90 (daily rate) and n (wells averaged)
    curves: pd.DataFrame      # one column per well, index t (months since peak), rate
    fits: dict                # "P10"/"P50"/"P90" -> DeclineFit of that profile
    eurs: pd.Series           # each well's EUR (its own hyperbolic fit), main product units
    ratio: float              # median lifetime ratio of the other stream (GOR or yield)
    names: dict = field(default_factory=dict)  # well id -> name

    @property
    def n_wells(self):
        return self.curves.shape[1]


def _hist(t, q):
    """A history table that fit_arps accepts, from months t and rates q."""
    t = np.asarray(t, int)
    return pd.DataFrame({"t": t, "q": np.asarray(q, float),
                         "date": [pd.Timestamp("2000-01-01") + pd.DateOffset(months=int(m)) for m in t]})


def build_type_curve(wells, product="oil", min_months=12, min_share=0.3, q_limit=5.0):
    """Type curve from monthly production of several wells (rrc.clean_production layout).

    min_months : wells need this many producing months after peak to count
    min_share  : the profile stops when fewer than this share of the wells
                 (and fewer than 3) still report, so a few survivors don't set the tail
    q_limit    : forecast cutoff used for each well's EUR
    """
    p = PRODUCTS[product]
    curves, eurs, ratios, names = {}, {}, [], {}
    for well_id, df in wells.groupby("lease_no"):
        df = df.sort_values("date")
        if not (df[p["rate"]] > 0).any():
            continue
        hist = prepare_history(df, rate=p["rate"])
        if len(hist) < min_months:
            continue
        curves[well_id] = hist.set_index("t")["q"]
        names[well_id] = str(df["lease_name"].iloc[0])
        try:
            fit = fit_arps(hist, "hyperbolic")
            eurs[well_id] = eur(df, forecast(fit, hist, q_limit=q_limit), p["volume"])
        except Exception:  # a well that won't fit still counts in the profile
            pass
        ratios.append(secondary_ratio(df, product, months=None))
    if len(curves) < 3:
        raise ValueError(f"Only {len(curves)} wells have {min_months}+ months after peak; a type curve "
                         "needs at least 3.")

    mat = pd.DataFrame(curves).sort_index()
    n = mat.notna().sum(axis=1)
    keep = n >= max(3, int(np.ceil(min_share * mat.shape[1])))
    mat_k = mat[keep]
    profile = pd.DataFrame({
        "t": mat_k.index.to_numpy(int),
        "p10": np.nanpercentile(mat_k, 90, axis=1),
        "p50": np.nanpercentile(mat_k, 50, axis=1),
        "p90": np.nanpercentile(mat_k, 10, axis=1),
        "n": n[keep].to_numpy(int),
    })
    fits = {name: fit_arps(_hist(profile["t"], profile[col]), "hyperbolic")
            for name, col in [("P10", "p10"), ("P50", "p50"), ("P90", "p90")]}
    return TypeCurve(profile, mat, fits, pd.Series(eurs, dtype=float),
                     float(np.median(ratios)) if ratios else 0.0, names)


def pick_candidates(leases, kind="Oil", n=20, seed=1):
    """Wells to sample from a PDQ field list (rrc.parse_field_leases rows).

    Keeps wells that produced in the listed period, skips disposal, brine and
    injection wells, and samples from the recent-but-not-newest lease numbers
    (RRC numbers increase over time), so each well has a peak and some decline.
    Returns up to 2n candidates, since some won't have enough history.
    """
    main = "oil" if kind == "Oil" else "gas"
    keep = [w for w in leases if w["kind"] == kind and w[main] > 0 and not NOT_PRODUCERS.search(w["name"])]
    keep.sort(key=lambda w: int(w["lease_no"]))
    band = keep[int(len(keep) * 0.55): int(len(keep) * 0.92)] or keep
    return random.Random(seed).sample(band, min(len(band), 2 * n))


def fetch_field_wells(field_no, kind="Oil", n=20, min_months=24, progress=None):
    """Monthly production for up to n wells in one Texas field, from the RRC.

    Returns (production DataFrame, field name). progress(i, total, label) is
    called before each fetch, for a progress bar.
    """
    end = pd.Timestamp.today()
    start = (end - pd.DateOffset(months=12)).strftime("%Y-%m")
    session = PDQSession(str(field_no), None, start=start, end=end.strftime("%Y-%m"), kind=kind)
    candidates = pick_candidates(session.field_leases(), kind=kind, n=n)
    rate = "bopd" if kind == "Oil" else "gas_mcfd"
    frames = []
    for i, w in enumerate(candidates):
        if len(frames) >= n:
            break
        if progress:
            progress(i, len(candidates), w["name"])
        try:
            df = clean_production(fetch_lease(w["lease_no"], w["district"], kind=kind))
            if (df[rate] > 0).sum() >= min_months and len(prepare_history(df, rate=rate)) >= min_months:
                frames.append(df)
        except Exception:  # skip wells PDQ can't return; others will fill in
            continue
    if not frames:
        raise ValueError(f"No wells with {min_months}+ months of decline found in field {field_no}.")
    return pd.concat(frames, ignore_index=True), session.field_name
