"""Download monthly production for a set of Spraberry Trend leases from the RRC.

Run from the project folder with:  python fetch_data.py
Writes data/spraberry_leases.csv (a few hundred rows, small enough to keep in git).
"""

from pathlib import Path

import pandas as pd

from production.rrc import PDQSession

ROOT = Path(__file__).parent

FIELD_NO, FIELD_NAME = "85280300", "SPRABERRY (TREND AREA)"  # District 08, Midland Basin

# Horizontal-well leases with a lettered name are usually one well per lease,
# which avoids splitting lease-level oil between wells. Chosen for 4-8 years of
# continuous history with a clear peak and decline.
LEASES = [
    (50678, "ACT A"),
    (51828, "ANOTHER NEWDGE 06-43"),  # has shut-in (zero) months - tests cleaning
    (52400, "ALLDALE SN"),
    (53612, "AGGIE THE BULLDOG 39-46 B"),
    (54464, "ABEL 18G"),
    (54537, "ALLEN-NEWMAN 12A"),
    (54657, "ABBIE LAINE 30E"),
    (54786, "ADAMS 43-6 C"),
]

session = PDQSession(FIELD_NO, FIELD_NAME, start="2015-01", end="2026-06")
records = []
for lease_no, name in LEASES:
    rows = session.lease_monthly(lease_no, name)
    print(f"{name:<28} {len(rows):>4} months")
    records += rows

out = ROOT / "data" / "spraberry_leases.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(records).to_csv(out, index=False)
print(f"Saved {out}")

# Historical monthly WTI spot price from the St. Louis Fed (FRED), public, no key needed.
# Used to price the months a well has already produced.
wti = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILWTICO")
wti.to_csv(ROOT / "data" / "wti_monthly.csv", index=False)
print(f"Saved {ROOT / 'data' / 'wti_monthly.csv'} ({len(wti)} months)")
