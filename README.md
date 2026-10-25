# PETE Production Analyzer

An upstream appraisal dashboard that takes a well from rock to dollars. Upload a
LAS well log and it works out:

1. **Well log** → net pay, porosity, water saturation (via the
   [PETE Python Log Reader](../pete-python-log-reader))
2. **Volumetrics** → original oil in place (OOIP)
3. **Production forecast** → EUR, in one of three modes:

   | Mode | Production data | How the forecast is built |
   |---|---|---|
   | **Prospect: typical assumptions** | None | EUR = OOIP × assumed recovery factor, with a typical decline shape for the well type |
   | **Prospect: decline from an analog** | A *similar* well's lease | Same EUR, but the decline shape (b, first-year decline) and gas-oil ratio come from the analog's actual production. Your log sets the size; the analog sets the shape. |
   | **Producing well** | The *same* well's lease | Arps decline fitted to the well's own history. The recovery factor (EUR ÷ OOIP) becomes a cross-check between the log and the production. |

   Leases can be fetched live from the Texas Railroad Commission by lease number and
   district, or picked from the bundled Spraberry examples.
4. **Economics** → DCF: NPV, IRR, payout, breakeven oil price, and for producing
   wells, the remaining (PDP) value against an asking price

Starting assumptions (drainage area, recovery factor, decline shape, capex, LOE)
come from the chosen well type, vertical conventional or horizontal tight oil, and
are all editable.

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

## Tests

```bash
python -m pytest
```

Unit tests check the decline math against known Arps curves, the cash flow
against hand calculations, the recovery factor verdicts, and the RRC parsing.
`tests/test_app.py` runs the whole dashboard headlessly in every mode. No test
uses the network.

## Data

- `data/spraberry_leases.csv`: monthly oil and casinghead gas for 8 horizontal-well
  leases in the Spraberry (Trend Area) field, District 08 (Midland Basin), from the
  RRC [Production Data Query](https://webapps.rrc.texas.gov/PDQ/). Refresh with
  `python fetch_data.py`.
- `data/wti_monthly.csv`: monthly WTI spot price from FRED (series MCOILWTICO).
- The bundled well log is a 12 ft synthetic test well from the log reader, not a log
  from these leases. Upload a real LAS file in the dashboard sidebar.

## Method

**Log-only type curve** (`production/decline.type_curve`)
- Target EUR = OOIP × recovery factor. The decline shape (b, first-year decline)
  comes from well-type defaults or a hyperbolic fit to an analog lease; the initial
  rate is solved so the forecast recovers exactly the target EUR before falling
  below the cutoff rate.
- An analog can't supply a recovery factor: that needs the analog's own oil in
  place, i.e. its log. The recovery factor stays a user input.
- Analog gas-oil ratio is its lifetime average; late-life GOR rises as reservoir
  pressure falls and would overstate a new well's gas.

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
- Lease fetching supports oil leases only; gas wells are reported separately by
  the RRC.
- Prices are flat for the forecast; there is no forward strip.
