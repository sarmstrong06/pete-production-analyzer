# PETE Production Analyzer

An upstream appraisal dashboard that takes a well from rock to dollars:

1. **Well log** → net pay, porosity, water saturation (via the
   [PETE Python Log Reader](../pete-python-log-reader))
2. **Volumetrics** → original oil in place (OOIP)
3. **Decline curve** → Arps fit to real Texas Railroad Commission production, EUR
4. **Cross-check** → recovery factor = EUR ÷ OOIP, flagged if implausible
5. **Economics** → DCF: full-cycle NPV, IRR, payout, breakeven oil price, and the
   remaining (PDP) value of the producing well against an asking price

## Setup (Ubuntu / WSL)

Uses the same virtual environment as the log reader. Run from this folder:

```bash
source ~/.venvs/well-log/bin/activate
pip install -r requirements.txt
```

This also installs the log reader as an editable package, so `import welllog` works
and picks up changes to the log reader immediately.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Data

- `data/spraberry_leases.csv`: monthly oil and casinghead gas for 8 horizontal-well
  leases in the Spraberry (Trend Area) field, District 08 (Midland Basin), from the
  RRC [Production Data Query](https://webapps.rrc.texas.gov/PDQ/). Refresh with
  `python fetch_data.py`.
- `data/wti_monthly.csv`: monthly WTI spot price from FRED (series MCOILWTICO).
- The bundled well log is a 12 ft synthetic test well from the log reader, not a log
  from these leases. Upload a real LAS file in the dashboard sidebar.

## Method

**Decline curves** (`production/decline.py`)
- Fits start at peak rate by default; the fit window can be moved later to skip a
  workover or refrac.
- Zero and unreported months are excluded. Fits use log(rate) with a robust
  (soft L1) loss, so partial-downtime months don't drag the curve down.
- Hyperbolic forecasts switch to exponential at a terminal decline (default 8%/yr),
  so b > 1 can't forecast infinite reserves.
- EUR = actual cumulative oil + forecast until the rate falls below 5 bbl/day.

**Economics** (`production/economics.py`)
- Monthly cash flow: net revenue (after royalty) − severance and ad valorem taxes −
  fixed and variable LOE − capex, discounted mid-month.
- Production stops at the economic limit (last month with positive operating cash).
- Full-cycle economics price past months at historical WTI and forecast months at a
  flat price (default: last 12 months' average WTI).
- Default assumptions (25% royalty, Texas severance 4.6% oil / 7.5% gas, $8,000/month
  + $4/bbl LOE, $8M capex) are typical Permian values, all editable in the dashboard.

**Cross-check** (`production/appraisal.py`)
- Typical tight-oil recovery factor is 3–15% of OOIP; outside 1–30% the log and
  the production data almost certainly don't describe the same rock volume.

## Limitations

- RRC reports oil by **lease**. The leases chosen are lettered horizontal-well
  leases, which are usually one well each, but this isn't verified per lease.
- PDQ lease data has no API numbers, so logs and leases are paired by hand rather
  than joined on API.
- Prices are flat for the forecast; there is no forward strip.
