"""Offline tests for the RRC parsing and cleaning (no network). The fixtures are
trimmed copies of real RRC pages."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from production.rrc import (_rows, clean_production, lease_kind, normalize_api, parse_field_leases,
                            parse_lease_page, parse_wellbore_results)

FIXTURES = Path(__file__).parent / "fixtures"


def test_rows_reads_innermost_rows_of_nested_tables():
    html = """
    <table><tr><td>
      <table>
        <tr><td>Jan 2024</td><td>1,234</td><td>x</td></tr>
        <tr><td>Feb&nbsp;2024</td><td><b>NO RPT</b></td></tr>
      </table>
    </td></tr></table>"""
    rows = [r for r in _rows(html) if r]
    assert rows == [["Jan 2024", "1,234", "x"], ["Feb 2024", "NO RPT"]]


def raw(oil, gas, operator):
    n = len(oil)
    return pd.DataFrame({
        "lease_no": [1] * n, "lease_name": ["TEST"] * n, "field": ["F"] * n, "field_no": ["1"] * n,
        "date": [f"2024-{m:02d}" for m in range(1, n + 1)],
        "oil_bbl": oil, "gas_mcf": gas, "operator": operator,
    })


def test_clean_production():
    df = clean_production(raw(
        oil=["3,100", "NO RPT", "0", "2,900", "NO RPT", "NO RPT"],
        gas=["6,000", "NO RPT", "0", "5,800", "NO RPT", "NO RPT"],
        operator=["ACME OIL", "", "", "", "", ""],
    ))
    assert len(df) == 4                                  # trailing unreported months dropped
    assert df["oil_bbl"].iloc[0] == 3100                 # thousands separators removed
    assert np.isnan(df["oil_bbl"].iloc[1])               # NO RPT is missing, not zero
    assert df["oil_bbl"].iloc[2] == 0                    # a reported zero stays zero
    assert (df["operator"] == "ACME OIL").all()          # operator filled forward
    assert (df["kind"] == "Oil").all()                   # old files without a kind are oil
    assert df["bopd"].iloc[0] == 3100 / 31
    assert df["gas_mcfd"].iloc[3] == 5800 / 30           # April has 30 days


@pytest.mark.parametrize("api,expected", [
    ("42-227-40326", "22740326"), ("4222740326", "22740326"), ("42227403260000", "22740326"),
    ("227-40326", "22740326"), ("30-015-12345", None), ("", None), (None, None), ("12345", None),
])
def test_normalize_api(api, expected):
    assert normalize_api(api) == expected


def test_lease_kind():
    assert lease_kind("54464") == "Oil"
    assert lease_kind("044426") == "Gas"


def test_parse_wellbore_results():
    rows = parse_wellbore_results((FIXTURES / "wellbore_results.html").read_text())
    assert len(rows) == 2
    assert rows[0] == {
        "api": "22740326", "district": "08", "lease_no": "54464", "kind": "Oil",
        "lease_name": "ABEL 18G", "well_no": "4HJ", "field": "SPRABERRY (TREND AREA)",
        "field_no": "85280300", "operator": "OVINTIV USA INC.", "county": "HOWARD",
    }
    assert rows[1]["well_no"] == "5HK"


def test_parse_field_lease_lists():
    oil = parse_field_leases((FIXTURES / "pdq_field_oil.html").read_text())
    assert len(oil) == 20
    assert oil[1] == {"name": "'M' FEE-POWELL UNIT", "lease_no": "53036", "district": "08", "kind": "Oil",
                      "oil": 334.0, "gas": 850.0}
    gas = parse_field_leases((FIXTURES / "pdq_field_gas.html").read_text())
    assert len(gas) == 20
    assert gas[1] == {"name": "1ST BAPTIST CHURCH", "lease_no": "252679", "district": "09", "kind": "Gas",
                      "oil": 0.0, "gas": 29201.0}


def test_parse_gas_well_page():
    """Gas wells report gas in the first column and condensate in the third."""
    recs = parse_lease_page((FIXTURES / "pdq_gas_well.html").read_text(), "044426", "DOBBINS, ETTA T.", "Gas")
    assert len(recs) == 6
    assert recs[0]["date"] == "2015-01" and recs[0]["kind"] == "Gas"
    assert recs[0]["gas_mcf"] == "1,059" and recs[0]["oil_bbl"] == "0"
    assert recs[5]["oil_bbl"] == "0"   # Jun 2015: 0 produced (15 disposed from storage is ignored)
    assert recs[0]["field"] == "BOONSVILLE (BEND CONGL., GAS)"
    df = clean_production(recs)
    assert df["gas_mcfd"].iloc[0] == pytest.approx(1059 / 31)
