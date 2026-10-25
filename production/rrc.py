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


def lease_kind(lease_no):
    """'Oil' or 'Gas' from the RRC numbering: oil leases have 5-digit numbers,
    gas wells 6-digit IDs (often with a leading zero)."""
    return "Gas" if len(str(lease_no).strip()) >= 6 else "Oil"


def fetch_lease(lease_no, district, kind=None, start="1993-01", end=None, timeout=60):
    """Monthly production for any Texas oil lease or gas well, via PDQ's
    Specific Lease Query.

    lease_no : RRC oil lease number (5 digits) or gas well ID (6 digits), found on
               the lease sign, drilling permits, or by API number (lookup_api)
    district : RRC district name, e.g. "08" or "7C"
    kind     : "Oil" or "Gas"; guessed from the number of digits if not given
    Returns raw records in the same layout as the saved CSV; pass them through
    clean_production. Oil leases report oil and casinghead gas; gas wells report
    gas-well gas and condensate (stored as gas_mcf and oil_bbl). Raises
    ValueError if PDQ has no such lease.
    """
    kind = kind or lease_kind(lease_no)
    label = "oil lease" if kind == "Oil" else "gas well"
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    (sy, sm), (ey, em) = start.split("-"), end.split("-")
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (PETE Production Analyzer)"
    r = s.get(BASE + "/PDQ/quickLeaseReportBuilderAction.do", timeout=timeout)
    action = re.search(r'name="QuickLeaseSearchForm" method="post" action="([^"]+)"', r.text, re.I)
    if not action:
        raise RuntimeError("PDQ lease search page has changed; the fetcher needs updating.")
    r = s.post(BASE + action.group(1), timeout=timeout, data={
        "wellType": kind, "leaseNumber": str(lease_no), "district": DISTRICT_CODES[district],
        "startMonth": sm, "startYear": sy, "endMonth": em, "endYear": ey, "submit": "Submit"})
    text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", r.text)))
    header = re.search(r"Lease Name:\s*(.*?),\s*Lease No", text)
    if "No Matches Found" in text or not header or not header.group(1).strip():
        raise ValueError(f"No {label} {lease_no} found in District {district}.")
    lease_name = header.group(1).strip()
    well_no = re.search(r"Lease No:\s*\d+,\s*Well No:\s*(.*?)\s+District", text)
    if well_no and well_no.group(1).strip():  # gas wells: several can share a lease name
        lease_name += f" #{well_no.group(1).replace(' ', '')}"
    page = s.get(BASE + "/PDQ/changePageViewAction.do", params={"pagesize": "500"}, timeout=timeout).text
    records = parse_lease_page(page, lease_no, lease_name, kind)
    if not records:
        raise ValueError(f"{label.capitalize()} {lease_no} (District {district}) has no monthly production on PDQ.")
    return records


def parse_lease_page(page_html, lease_no, lease_name, kind="Oil"):
    """Monthly rows of a PDQ lease production page. Columns are the same for
    both kinds: date, liquid or gas production, disposition, the other stream's
    production, disposition, operator, operator no., field, field no."""
    records, field, field_no = [], "", ""
    for c in _rows(page_html):
        if len(c) >= 5 and re.fullmatch(r"[A-Z][a-z]{2} \d{4}", c[0]):
            mon, year = c[0].split()
            if len(c) > 8 and c[7]:
                field, field_no = c[7], c[8]
            oil, gas = (c[1], c[3]) if kind == "Oil" else (c[3], c[1])
            records.append({
                "lease_no": int(lease_no), "lease_name": lease_name, "kind": kind,
                "field": field, "field_no": field_no,
                "date": f"{year}-{MONTHS[mon]:02d}",
                "oil_bbl": oil, "gas_mcf": gas,
                "operator": c[5] if len(c) > 5 else "",
            })
    return records


def normalize_api(api):
    """Texas API number as 8 digits (3-digit county + 5-digit well), or None.

    Accepts the forms found in LAS headers: "42-227-40326", "4222740326",
    "42227403260000" (14-digit, with sidetrack/event codes), "227-40326".
    Texas is state code 42; other states return None.
    """
    digits = re.sub(r"\D", "", str(api or ""))
    if len(digits) in (10, 12, 14):
        if not digits.startswith("42"):
            return None
        digits = digits[2:10]
    return digits if len(digits) == 8 else None


def wellbore_query(timeout=60, **criteria):
    """Search the RRC Wellbore Query (wells by API, lease, field, county...).

    criteria : form fields without the "searchArgs." prefix, e.g.
               apiNoPrefixArg="227", apiNoSuffixArg="40326"
    Returns parsed rows (see parse_wellbore_results), up to the site's limit.
    """
    base = "https://webapps2.rrc.texas.gov/EWA/"
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (PETE Production Analyzer)"
    r = s.get(base + "wellboreQueryAction.do", timeout=timeout)
    action = re.search(r'name="wellboreQueryActionForm"[^>]*action="([^"]+)"', r.text) \
        or re.search(r'action="(/EWA/wellboreQueryAction[^"]+)"', r.text)
    if not action:
        raise RuntimeError("RRC Wellbore Query page has changed; the lookup needs updating.")
    fields = {"fieldNumbersArg": "", "operatorNumbersArg": "", "leaseTypeArg": "",
              "districtCodeArg": "None Selected", "leaseNumberArg": "", "wellTypeArg": "None Selected",
              "countyCodeArg": "None Selected", "drillingPermitArg": "", "apiNoPrefixArg": "",
              "apiNoSuffixArg": "", "scheduleTypeArg": "Both", **criteria}
    data = {"methodToCall": "search", "pager.pageSize": "-1",
            **{f"searchArgs.{k}": v for k, v in fields.items()}}
    r = s.post(requests.compat.urljoin(base, action.group(1)), timeout=timeout, data=data)
    return parse_wellbore_results(r.text)


def lookup_api(api, timeout=60):
    """Find the RRC lease for a well from its API number, via the RRC Wellbore Query.

    Returns a list of dicts (usually one) with api, district, lease_no, kind
    ("Oil" or "Gas"), lease_name, well_no, field, field_no, operator, county.
    Empty if the RRC has no such wellbore. Raises ValueError if the API number
    isn't a valid Texas one.
    """
    api8 = normalize_api(api)
    if not api8:
        raise ValueError(f"{api!r} is not a Texas API number (42-CCC-WWWWW).")
    return wellbore_query(timeout, apiNoPrefixArg=api8[:3], apiNoSuffixArg=api8[3:])


def lease_wells(lease_no, district, kind="Oil", timeout=60):
    """Wellbores on one RRC lease (an oil lease can have several; a gas well ID is one well)."""
    return wellbore_query(timeout, leaseTypeArg="O" if kind == "Oil" else "G",
                          districtCodeArg=district, leaseNumberArg=str(lease_no))


def parse_wellbore_results(page_html):
    """Rows of an RRC Wellbore Query results page (see lookup_api)."""
    starts = [m.start() for m in re.finditer(r'<a href="leaseDetailAction\.do\?searchType=apiNo', page_html)]
    results = []
    for i, start in enumerate(starts):
        seg = page_html[start: starts[i + 1] if i + 1 < len(starts) else len(page_html)]
        link = re.search(r"apiNo=(\d+)&(?:amp;)?distCode=(\w+)&(?:amp;)?leaseNo=(\d+)", seg)
        if not link:
            continue
        # Drop the per-row dropdown and the API cell's nested table. The lease
        # number cell is followed by an empty spacer cell, then: lease name,
        # well no, field, operator, county
        seg = re.sub(r"<select.*?</select>", "", seg, flags=re.S)
        cells = _cells(seg.split("</table>", 1)[-1])
        li = cells.index(link.group(3)) if link.group(3) in cells else len(cells)
        get = lambda k: cells[li + k] if len(cells) > li + k else ""
        field_no = re.search(r"field=(\d+)", seg)
        results.append({
            "api": link.group(1), "district": link.group(2), "lease_no": link.group(3),
            "kind": lease_kind(link.group(3)), "lease_name": get(2), "well_no": get(3), "field": get(4),
            "field_no": field_no.group(1) if field_no else "",
            "operator": get(5), "county": get(6),
        })
    return results


def clean_production(df):
    """Clean raw production records for analysis.

    - "NO RPT" (operator filed no report) becomes missing, not zero.
    - Trailing unreported months are dropped: they're months the RRC hasn't
      received reports for yet, not months the lease stopped producing.
    - Operator name is only printed when it changes, so it is filled forward.
    - Adds days-in-month and daily rates (BOPD, MCF/D).
    """
    df = pd.DataFrame(df).astype({"oil_bbl": str, "gas_mcf": str, "operator": str})
    if "kind" not in df:
        df["kind"] = "Oil"
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    for col in ("oil_bbl", "gas_mcf"):
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    df["operator"] = df["operator"].replace({"": pd.NA, "nan": pd.NA})
    df = df.sort_values(["lease_no", "date"]).reset_index(drop=True)
    df["operator"] = df.groupby("lease_no")["operator"].ffill()
    reported = df["oil_bbl"].notna() | df["gas_mcf"].notna()
    last_reported = df[reported].groupby("lease_no")["date"].max()
    df = df[df["date"] <= df["lease_no"].map(last_reported)].reset_index(drop=True)
    df["days"] = df["date"].dt.days_in_month
    df["bopd"] = df["oil_bbl"] / df["days"]
    df["gas_mcfd"] = df["gas_mcf"] / df["days"]
    return df


def load_production(path):
    """Load a saved production CSV and clean it (see clean_production)."""
    return clean_production(pd.read_csv(path, dtype={"oil_bbl": str, "gas_mcf": str, "operator": str,
                                                     "field_no": str}))
