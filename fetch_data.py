"""Download the example production and price data from public sources.

Run from the project folder with:  python fetch_data.py
Writes (all small enough to keep in git):
- data/spraberry_leases.csv  : 8 Spraberry Trend oil leases (RRC PDQ)
- data/barnett_gas_wells.csv : 8 Barnett Shale gas wells (RRC PDQ)
- data/wti_monthly.csv       : WTI spot price (FRED)
- data/henry_hub_monthly.csv : Henry Hub spot gas price (FRED)
- data/raw/all_kansas_leases.zip : KGS index of Kansas leases and their wells'
  API numbers, used to find a Kansas well's lease (4 MB; not kept in git)
"""

from pathlib import Path

import pandas as pd
import requests

from production.rrc import PDQSession, fetch_lease

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ---- Oil: Spraberry (Trend Area), District 08, Midland Basin
FIELD_NO, FIELD_NAME = "85280300", "SPRABERRY (TREND AREA)"

# Horizontal-well leases with a lettered name are often one well per lease,
# which avoids splitting lease-level oil between wells (ABEL 18G has two).
# Chosen for 4-8 years of continuous history with a clear peak and decline.
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
pd.DataFrame(records).assign(kind="Oil").to_csv(DATA / "spraberry_leases.csv", index=False)
print(f"Saved {DATA / 'spraberry_leases.csv'}")

# ---- Gas: Newark, East (Barnett Shale), District 09, Tarrant County
# Gas wells are reported individually (one RRC gas well ID per well), so there
# is no lease allocation problem. Chosen for long, nearly gap-free histories.
GAS_WELLS = ["196851", "196304", "196303", "197314", "199924", "199799", "198592", "196226"]

records = []
for well_id in GAS_WELLS:
    rows = fetch_lease(well_id, "09", kind="Gas")
    print(f"{rows[0]['lease_name']:<28} {len(rows):>4} months")
    records += rows
pd.DataFrame(records).to_csv(DATA / "barnett_gas_wells.csv", index=False)
print(f"Saved {DATA / 'barnett_gas_wells.csv'}")

# ---- Historical monthly prices from the St. Louis Fed (FRED), public, no key needed.
# Used to price the months a well has already produced.
for series, name in [("MCOILWTICO", "wti_monthly.csv"), ("MHHNGSP", "henry_hub_monthly.csv")]:
    p = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    p.to_csv(DATA / name, index=False)
    print(f"Saved {DATA / name} ({len(p)} months)")

# ---- Kansas lease index from the Kansas Geological Survey (public download).
# Links each lease to its wells' API numbers; the dashboard uses it to find a
# Kansas well's lease. Kept in data/raw/, which git ignores.
(DATA / "raw").mkdir(exist_ok=True)
r = requests.get("https://www.kgs.ku.edu/PRS/Ora_Archive/all_kansas_leases.zip", timeout=300)
r.raise_for_status()
(DATA / "raw" / "all_kansas_leases.zip").write_bytes(r.content)
print(f"Saved {DATA / 'raw' / 'all_kansas_leases.zip'} ({len(r.content) / 1e6:.1f} MB)")
