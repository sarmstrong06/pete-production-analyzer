"""Offline tests for the Kansas (KGS) parsing. The data below is made up in the
KGS file format; real KGS monthly volumes are licensed and not stored here."""

import pandas as pd
import pytest

from production.kgs import parse_month_file, parse_well_page, split_api, spread_sales, wells_on_lease

HEADER = ('"LEASE_KID","LEASE","DOR_CODE","API_NUMBER","FIELD","PRODUCING_ZONE","OPERATOR","COUNTY",'
          '"TOWNSHIP","TWN_DIR","RANGE","RANGE_DIR","SECTION","SPOT","LATITUDE","LONGITUDE",'
          '"MONTH-YEAR","PRODUCT","WELLS","PRODUCTION"')


def row(month_year, bbl, wells=1, product="O"):
    return (f'"1000000001","Test 1","999999","15-001-00001 ","TEST FIELD","","Test Oil Co","Allen",'
            f'"1","S","1","E","1","","38.0","-98.0","{month_year}","{product}","{wells}","{bbl}"')


SAMPLE = "\n".join([HEADER, row("0-2020", 900), row("5-2020", 300), row("9-2020", 300),
                    row("3-2021", 160), row("10-2021", 200, wells=2), row("2-2022", 50, wells=2),
                    row("6-2021", 999, product="G")]) + "\n"


@pytest.mark.parametrize("api,expected", [
    ("15-167-24,013-00-00", ("15", "167", "24013")), ("42-227-40326", ("42", "227", "40326")),
    ("1516723617", ("15", "167", "23617")), ("167-23617", None), ("", None),
])
def test_split_api(api, expected):
    assert split_api(api) == expected


def test_parse_month_file_keeps_monthly_oil_rows():
    sales = parse_month_file(SAMPLE)
    assert len(sales) == 5                       # annual (month 0) and gas rows dropped
    assert sales["date"].iloc[0] == pd.Timestamp("2020-05-01")
    assert sales["sales_bbl"].sum() == 1010
    assert wells_on_lease(sales) == 2


def test_spread_sales_keeps_yearly_totals():
    m = spread_sales(parse_month_file(SAMPLE))
    assert m["date"].iloc[0] == pd.Timestamp("2020-05-01") and m["date"].iloc[-1] == pd.Timestamp("2022-02-01")
    by_year = m.groupby(m["date"].dt.year)["oil_bbl"]
    assert by_year.sum().tolist() == pytest.approx([600, 360, 50])
    assert by_year.size().tolist() == [8, 12, 2]          # May-Dec, all of 2021, Jan-Feb
    assert m["oil_bbl"].iloc[0] == pytest.approx(600 / 8)  # spread evenly
    assert (m["kind"] == "Oil").all() and m["gas_mcf"].isna().all()
    assert m["bopd"].iloc[0] == pytest.approx(75 / 31)


def test_parse_well_page():
    page = """<html><body>API: 15-001-00001 KID: 1 Lease: Test Well: 1 Original operator: X
    Current operator: Test Oil Co Field: Test Field Location: T1S County: Allen Well Type: OIL Status: Producing
    <a href="https://chasm.kgs.ku.edu/ords/oil.ogl5.MainLease?f_lc=1000000001">Oil production</a></body></html>"""
    w = parse_well_page(page)
    assert w["lease_kid"] == "1000000001" and w["api"] == "15-001-00001"
    assert w["lease_name"] == "Test" and w["well"] == "1" and w["field"] == "Test Field"
    assert w["operator"] == "Test Oil Co" and w["well_type"] == "OIL"
    assert parse_well_page("<html>dry hole, no lease</html>") is None
