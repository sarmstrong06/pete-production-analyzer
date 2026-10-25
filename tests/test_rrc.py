"""Offline tests for the RRC parsing and cleaning (no network)."""

import numpy as np
import pandas as pd

from production.rrc import _rows, clean_production


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
    assert df["bopd"].iloc[0] == 3100 / 31
    assert df["gas_mcfd"].iloc[3] == 5800 / 30           # April has 30 days
