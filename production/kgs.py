"""Kansas oil lease production from the Kansas Geological Survey (KGS).

Kansas wells are found by API number (state code 15) in the KGS lease index
(all_kansas_leases.zip, which lists each lease's wells), downloaded to
data/raw/ by fetch_data.py.

Two things differ from Texas:
- Kansas reports oil *sales* (tank-truck loads sold to a purchaser), not
  production. A small well sells a ~165 bbl load whenever its tank fills, so
  monthly volumes are lumpy, with empty months between loads. To get a
  production-like series, each year's sales are spread evenly over the months
  the lease was producing that year.
- Oil leases report no casinghead gas, so gas is missing (forecast gas = 0).

KGS licenses the monthly volumes from IHS for research use, not for
redistribution, so this data is fetched live and not saved in the repository.
"""

import io
import re
import zipfile
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://chasm.kgs.ku.edu/ords/"
UA = {"User-Agent": "Mozilla/5.0 (PETE Production Analyzer)"}


def split_api(api):
    """(state, county, well) codes from an API number in any common format, or None.

    "15-167-24,013-00-00" -> ("15", "167", "24013"); "42-227-40326" -> ("42", "227", "40326").
    """
    digits = re.sub(r"\D", "", str(api or ""))
    if len(digits) in (10, 12, 14):
        return digits[:2], digits[2:5], digits[5:10]
    return None


def _text(html):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def lookup_api(api, index_path=None, timeout=60):
    """Find a Kansas well's oil lease from its API number.

    Uses the KGS lease index (all_kansas_leases.zip, downloaded by
    fetch_data.py) when index_path exists; otherwise searches the KGS website
    (LAS file index, then the well page, which KGS may refuse to automated
    visitors). Returns a list of dicts with api, lease_kid, lease_name, field,
    operator, county, n_wells; empty if the well has no oil lease.
    """
    parts = split_api(api)
    if not parts or parts[0] != "15":
        raise ValueError(f"{api!r} is not a Kansas API number (15-CCC-WWWWW).")
    if index_path is not None and Path(index_path).exists():
        return lookup_in_index(api, index_path)
    _, county, well = parts
    s = requests.Session()
    s.headers.update(UA)
    r = s.post(BASE + "las.lasd5.SelectWells", timeout=timeout, data={
        "f_s": "", "f_t": "", "f_r": "", "f_l": "", "f_op": "", "f_st": "15",
        "f_c": str(int(county)), "f_api": well})
    kids = list(dict.fromkeys(re.findall(r"DisplayWell\?f_kid=(\d+)", r.text)))
    results = []
    for kid in kids:
        page = s.get(BASE + f"qualified.well_page.DisplayWell?f_kid={kid}", timeout=timeout).text
        found = parse_well_page(page)
        if found:
            results.append({"well_kid": kid, **found})
    return results


def lookup_in_index(api, index_path):
    """lookup_api against the KGS lease index file (zip of all_kansas_leases.txt, or the .txt)."""
    state, county, well = split_api(api)
    key = f"{state}-{county}-{well}"
    path = Path(index_path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
    else:
        text = path.read_text(errors="ignore")
    leases = pd.read_csv(io.StringIO(text), dtype=str)
    leases = leases[leases["API_NUMBER"].fillna("").str.contains(key, regex=False)
                    & (leases["PRODUCES"].fillna("") == "OIL")]
    return [{
        "api": key, "lease_kid": r["LEASE_KID"], "lease_name": r["LEASE"].strip(), "well": "",
        "field": "", "operator": (r["OPERATOR"] or "").strip(), "county": r["COUNTY"],
        "n_wells": len([a for a in str(r["API_NUMBER"]).split(",") if a.strip()]),
    } for _, r in leases.iterrows()]


def parse_well_page(page_html):
    """Lease and well details from a KGS well page, or None if it has no oil lease."""
    lease = re.search(r"oil\.ogl5\.MainLease\?f_lc=(\d+)", page_html)
    if not lease:
        return None
    t = _text(page_html)
    get = lambda pat: (m.group(1).strip() if (m := re.search(pat, t)) else "")
    return {
        "api": get(r"API: (15-\d{3}-\d{5})"), "lease_kid": lease.group(1),
        "lease_name": get(r"Lease: (.*?) Well:"), "well": get(r"Well: (.*?) Original operator:"),
        "field": get(r"Field: (.*?) Location:"), "operator": get(r"Current operator: (.*?) Field:"),
        "county": get(r"County: (\w+)"), "well_type": get(r"Well Type: (\S+)"),
    }


def fetch_lease(lease_kid, timeout=60):
    """Monthly oil sales for one Kansas lease, as raw KGS rows (see parse_month_file)."""
    s = requests.Session()
    s.headers.update(UA)
    r = s.get(BASE + f"oil.ogl5.MonthSave?f_lc={lease_kid}", timeout=timeout)
    link = re.search(r'href="([^"]*anon_blobber\.download\?p_file_name=[^"]+)"', r.text)
    if not link:
        raise ValueError(f"KGS has no monthly production for lease {lease_kid}.")
    return parse_month_file(s.get(requests.compat.urljoin(r.url, unescape(link.group(1))), timeout=timeout).text)


def parse_month_file(text):
    """Parse a KGS monthly lease file (comma-delimited, one row per month with sales)."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df = df[df["PRODUCT"].str.strip() == "O"]
    month, year = df["MONTH-YEAR"].str.split("-", expand=True).astype(int).T.values
    keep = month >= 1  # month 0 / -1 rows are annual totals and starting cumulatives
    df, month, year = df[keep], month[keep], year[keep]
    return pd.DataFrame({
        "lease_no": df["LEASE_KID"].astype(np.int64).values, "lease_name": df["LEASE"].str.strip().values,
        "field": df["FIELD"].str.strip().values, "operator": df["OPERATOR"].str.strip().values,
        "api": df["API_NUMBER"].str.strip().values,
        "date": pd.to_datetime({"year": year, "month": month, "day": 1}).values,
        "wells": pd.to_numeric(df["WELLS"], errors="coerce").values,
        "sales_bbl": pd.to_numeric(df["PRODUCTION"], errors="coerce").values,
    })


def spread_sales(sales):
    """Monthly production-like series from lumpy monthly sales.

    Each calendar year's sales are spread evenly over the months the lease was
    producing that year: from the first sale (first year) or January, through
    December or the last reported month (last year). Output has the same
    columns as rrc.clean_production, with kind "Oil" and gas missing.
    """
    sales = sales.sort_values("date")
    first, last = sales["date"].min(), sales["date"].max()
    months = pd.date_range(first, last, freq="MS")
    out = pd.DataFrame({"date": months})
    totals = sales.groupby(sales["date"].dt.year)["sales_bbl"].sum()
    n_in_year = out.groupby(out["date"].dt.year)["date"].transform("size")
    out["oil_bbl"] = out["date"].dt.year.map(totals).fillna(0).to_numpy() / n_in_year.to_numpy()
    out["gas_mcf"] = np.nan
    info = sales.iloc[-1]
    out["lease_no"], out["lease_name"], out["kind"] = info["lease_no"], info["lease_name"], "Oil"
    out["field"], out["field_no"], out["operator"] = info["field"], "", info["operator"]
    out["days"] = out["date"].dt.days_in_month
    out["bopd"] = out["oil_bbl"] / out["days"]
    out["gas_mcfd"] = np.nan
    return out


def wells_on_lease(sales):
    """Number of wells the lease reported in its most recent month."""
    w = sales.sort_values("date")["wells"].dropna()
    return int(w.iloc[-1]) if len(w) else 1
