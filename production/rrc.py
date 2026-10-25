"""Fetch monthly lease production from the Texas Railroad Commission's
Production Data Query (PDQ) website, and load the saved CSV.

PDQ has no download API. It is a website that keeps your search in a server-side
session, so fetching repeats the clicks a person would make: open a session,
pick the field, run the lease search, then open each lease's monthly table.
"""

import re
import time
from html import unescape
from urllib.parse import quote_plus

import pandas as pd
import requests

BASE = "https://webapps.rrc.texas.gov"
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _cells(row_html):
    """Text of each <td> in one table row."""
    tds = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S | re.I)
    return [re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", td))).strip() for td in tds]


def _rows(page_html):
    """Cells of every innermost table row on a page."""
    rows = re.findall(r"<tr[^>]*>((?:(?!<tr).)*?)</tr>", page_html, flags=re.S | re.I)
    return [_cells(r) for r in rows]


class PDQSession:
    """One search session on the PDQ website for a single field."""

    def __init__(self, field_no, field_name, start="2015-01", end="2026-06", pause=0.5):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (PETE Production Analyzer)"
        self.pause = pause  # be polite to a public server

        r = self._get("/PDQ/generalReportAction.do")
        sid = re.search(r"jsessionid=([^\"?&;]+)", r.text).group(1)
        url = lambda p: f"{BASE}/PDQ/{p};jsessionid={sid}"

        self.s.post(url("mainReportAction.do"), data={"viewType": "Lease", "submit": "Search For Field"})
        self.s.post(url("fieldSearchAction.do"), data={
            "fieldSearchCriteria": "exactMatch", "fieldSearchValue": field_no, "submit": "Search"})
        r = self.s.post(url("fieldSelectAction.do"), data={
            "selectedFieldValue": f"{field_no}^{field_name}", "submit": "Submit"})

        action = re.search(r'name="SearchCriteriaForm" method="POST" action="([^"]+)"', r.text).group(1)
        hidden = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text))
        (sy, sm), (ey, em) = start.split("-"), end.split("-")
        r = self.s.post(BASE + action, data={
            **hidden, "viewType": "Lease", "wellType": "Oil",
            "startMonth": sm, "startYear": sy, "endMonth": em, "endYear": ey,
            "district": "None Selected", "onShoreCounty": "None Selected",
            "offShoreArea": "None Selected", "fieldNameData": field_name,
            "fieldRegion": f"{field_name} {field_no}",
            "operatorNameData": "None Selected", "operatorValue": "None Selected",
            "submit": "Submit"})
        if "Lease Query Results" not in r.text:
            raise RuntimeError("PDQ search failed; the website may have changed or the session timed out.")
        self.field_no, self.field_name = field_no, field_name
        self._in_lease = False  # True after opening a lease; must go back up before the next

    def _get(self, path, **params):
        time.sleep(self.pause)
        return self.s.get(BASE + path, params=params)

    def lease_monthly(self, lease_no, lease_name, district_code="10"):
        """Monthly oil and casinghead gas for one oil lease, as raw text cells.

        district_code is PDQ's internal code (district 08 is "10" on the website).
        """
        if self._in_lease:
            self._get("/PDQ/drillUpAction.do", viewType="Lease")
        r = self._get("/PDQ/drillDownAction.do", ogCode="O", name=quote_plus(lease_name),
                      number=str(lease_no), districtNo=district_code, gasWellNumber="")
        if "Lease Production" not in r.text:
            raise RuntimeError(f"Could not open lease {lease_no} ({lease_name}) on PDQ.")
        self._in_lease = True
        page = self._get("/PDQ/changePageViewAction.do", pagesize="500").text

        records = []
        for c in _rows(page):
            if len(c) >= 5 and re.fullmatch(r"[A-Z][a-z]{2} \d{4}", c[0]):
                mon, year = c[0].split()
                records.append({
                    "lease_no": lease_no, "lease_name": lease_name,
                    "field": self.field_name, "field_no": self.field_no,
                    "date": f"{year}-{MONTHS[mon]:02d}",
                    "oil_bbl": c[1], "gas_mcf": c[3],
                    "operator": c[5] if len(c) > 5 else "",
                })
        return records


# PDQ's internal district codes (what the website's form sends) by RRC district name
DISTRICT_CODES = {"01": "01", "02": "02", "03": "03", "04": "04", "05": "05", "06": "06",
                  "6E": "07", "7B": "08", "7C": "09", "08": "10", "8A": "11", "09": "13", "10": "14"}


def fetch_lease(lease_no, district, start="1993-01", end=None, timeout=60):
    """Monthly production for any Texas oil lease, via PDQ's Specific Lease Query.

    lease_no : RRC oil lease number (found on the lease's sign, permits, or the
               RRC's online maps)
    district : RRC district name, e.g. "08" or "7C"
    Returns raw records in the same layout as the saved CSV; pass them through
    clean_production. Raises ValueError if PDQ has no such oil lease.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    (sy, sm), (ey, em) = start.split("-"), end.split("-")
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (PETE Production Analyzer)"
    r = s.get(BASE + "/PDQ/quickLeaseReportBuilderAction.do", timeout=timeout)
    action = re.search(r'name="QuickLeaseSearchForm" method="post" action="([^"]+)"', r.text, re.I)
    if not action:
        raise RuntimeError("PDQ lease search page has changed; the fetcher needs updating.")
    r = s.post(BASE + action.group(1), timeout=timeout, data={
        "wellType": "Oil", "leaseNumber": str(lease_no), "district": DISTRICT_CODES[district],
        "startMonth": sm, "startYear": sy, "endMonth": em, "endYear": ey, "submit": "Submit"})
    header = re.search(r"Lease Name:\s*(.*?),\s*Lease No", re.sub(r"<[^>]+>", " ", r.text))
    if not header:
        raise ValueError(f"No oil lease {lease_no} found in District {district}.")
    lease_name = re.sub(r"\s+", " ", unescape(header.group(1))).strip()
    page = s.get(BASE + "/PDQ/changePageViewAction.do", params={"pagesize": "500"}, timeout=timeout).text

    records, field, field_no = [], "", ""
    for c in _rows(page):
        if len(c) >= 5 and re.fullmatch(r"[A-Z][a-z]{2} \d{4}", c[0]):
            mon, year = c[0].split()
            if len(c) > 8 and c[7]:
                field, field_no = c[7], c[8]
            records.append({
                "lease_no": int(lease_no), "lease_name": lease_name,
                "field": field, "field_no": field_no,
                "date": f"{year}-{MONTHS[mon]:02d}",
                "oil_bbl": c[1], "gas_mcf": c[3],
                "operator": c[5] if len(c) > 5 else "",
            })
    if not records:
        raise ValueError(f"Lease {lease_no} (District {district}) has no monthly production on PDQ.")
    return records


def clean_production(df):
    """Clean raw production records for analysis.

    - "NO RPT" (operator filed no report) becomes missing, not zero.
    - Trailing unreported months are dropped: they're months the RRC hasn't
      received reports for yet, not months the lease stopped producing.
    - Operator name is only printed when it changes, so it is filled forward.
    - Adds days-in-month and daily rates (BOPD, MCF/D).
    """
    df = pd.DataFrame(df).astype({"oil_bbl": str, "gas_mcf": str, "operator": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    for col in ("oil_bbl", "gas_mcf"):
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    df["operator"] = df["operator"].replace({"": pd.NA, "nan": pd.NA})
    df = df.sort_values(["lease_no", "date"]).reset_index(drop=True)
    df["operator"] = df.groupby("lease_no")["operator"].ffill()
    last_reported = df[df["oil_bbl"].notna()].groupby("lease_no")["date"].max()
    df = df[df["date"] <= df["lease_no"].map(last_reported)].reset_index(drop=True)
    df["days"] = df["date"].dt.days_in_month
    df["bopd"] = df["oil_bbl"] / df["days"]
    df["gas_mcfd"] = df["gas_mcf"] / df["days"]
    return df


def load_production(path):
    """Load a saved production CSV and clean it (see clean_production)."""
    return clean_production(pd.read_csv(path, dtype={"oil_bbl": str, "gas_mcf": str, "operator": str}))
