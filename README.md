# PETE Production Analyzer

An upstream appraisal dashboard that takes an oil or gas well from rock to
dollars. Upload a LAS well log and it works out:

1. **Well log** → net pay, porosity, water saturation (via the
   [PETE Python Log Reader](../pete-python-log-reader))
2. **Volumetrics** → original oil in place (OOIP) or gas in place (OGIP)
3. **Production forecast** → EUR, in one of three modes:

   | Mode | Production data | How the forecast is built |
   |---|---|---|
   | **Prospect: typical assumptions** | None | EUR = volume in place × assumed recovery factor, with a typical decline shape for the well type |
   | **Prospect: decline from an analog** | A *similar* well | Same EUR, but the decline shape (b, first-year decline) and the other stream's ratio (gas-oil ratio or condensate yield) come from the analog's actual production. Your log sets the size; the analog sets the shape. |
   | **Producing well** | The *same* well | Arps decline fitted to the well's own history. The recovery factor (EUR ÷ volume in place) becomes a cross-check between the log and the production. |

   Production is found from the log's **API number**: Texas wells (42-...) from
   the Railroad Commission, Kansas oil wells (15-...) from the Kansas Geological
   Survey. Texas wells can also be fetched by RRC lease number / gas well ID, or
   picked from the bundled examples (Spraberry oil leases, Barnett Shale gas wells).
4. **Economics** → DCF: NPV, IRR, payout, breakeven price (WTI for oil wells,
   Henry Hub for gas wells), and for producing wells, the remaining (PDP) value
   against an asking price. Forecast prices come from a flat, escalating, or
   custom yearly price deck.

Four well types set the starting assumptions (drainage area, recovery factor
range, decline shape, costs): vertical conventional oil, horizontal tight oil,
vertical conventional gas, and horizontal shale gas. All are editable.

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
against hand calculations, the IRR and breakeven solvers, price decks, the
recovery factor verdicts, and the RRC page parsing (against saved copies of
real RRC pages in `tests/fixtures/`). `tests/test_app.py` runs the whole
dashboard headlessly for every well type and mode. No test uses the network.

## Data

- `data/spraberry_leases.csv`: monthly oil and casinghead gas for 8 horizontal-well
  leases in the Spraberry (Trend Area) field, District 08 (Midland Basin).
- `data/barnett_gas_wells.csv`: monthly gas and condensate for 8 gas wells in the
  Newark, East (Barnett Shale) field, District 09, Tarrant County, with 20+ years
  of history each.
- `data/wti_monthly.csv`, `data/henry_hub_monthly.csv`: monthly WTI and Henry Hub
  spot prices from FRED (series MCOILWTICO, MHHNGSP).
- Refresh all of these with `python fetch_data.py`. Production comes from the RRC
  [Production Data Query](https://webapps.rrc.texas.gov/PDQ/); API lookups use the
  RRC [Wellbore Query](https://webapps2.rrc.texas.gov/EWA/wellboreQueryAction.do).
- `data/logs/nuss_1.las`: a real log, Nuss 1 (API 15-009-25378), Barton County,
  Kansas, courtesy of the Kansas Corporation Commission via the KGS. See the
  worked example below.
- The synthetic test well is the log reader's 12 ft sample, for testing only.

## Worked example: Nuss 1, Kansas

A real well, end to end. Pick **Real example: Nuss 1, Kansas** in the sidebar,
well type **Vertical oil**, mode **Producing well**.

Nuss 1 is the discovery well of the Galatia East field (drilled January 2010),
producing from the Lansing-Kansas City limestones. The zone (3,106-3,342 ft) and
lithology come from the KGS formation tops; the production is found from the
log's API number.

| Step | Result |
|---|---|
| Log, Lansing-Kansas City, limestone matrix | 23 ft net pay, 17.8% porosity, 22.9% water saturation |
| Volumetrics, 40 acres, Bo 1.3 | 754 Mbbl oil in place |
| Production (KGS, by API number) | 180 Mbbl produced 2010-2026; EUR 203 Mbbl |
| Cross-check | Recovery factor 27%, just above the typical 10-25% for a vertical conventional well |

The cross-check says "worth a second look": a 27% recovery factor suggests the well
drains more than the assumed 40 acres (or the pay is thicker than the cutoffs
count). Raising the drainage area to about 50 acres brings it into the typical
range. This is the kind of reconciliation between log and production the
dashboard is built for.

## Method

**Evaluating the log** (`app.py`, using the log reader)
- A zone window limits the evaluation to the producing formation. Whole-well logs
  contain salt, anhydrite and other zones that can falsely pass the pay cutoffs.
- Lithology sets the matrix density for density porosity: 2.65 g/cc sandstone,
  2.71 limestone, 2.87 dolomite.

**Matching a log to its production** (`production/rrc.py`)
- The LAS header's `API` (or `UWI`) number is normalized to the 8-digit Texas form
  (county + well) and looked up in the RRC Wellbore Query, which returns the well's
  district and lease number (5 digits, oil) or gas well ID (6 digits, gas).
- Oil is reported per lease. If the lease has several wellbores, the dashboard says
  so and divides lease volumes equally between them (editable).
- Kansas API numbers (state 15) go to the KGS instead (`production/kgs.py`): the
  KGS LAS index links the well to its oil lease. Kansas reports oil **sales**
  (truckloads sold), not production, so monthly volumes are lumpy; each year's
  sales are spread evenly over that year's producing months before fitting.
  Kansas oil leases report no casinghead gas.

**Log-only type curve** (`production/decline.type_curve`)
- Target EUR = volume in place × recovery factor. The decline shape (b, first-year
  decline) comes from well-type defaults or a hyperbolic fit to an analog; the
  initial rate is solved so the forecast recovers exactly the target EUR before
  falling below the cutoff rate.
- An analog can't supply a recovery factor: that needs the analog's own volume in
  place, i.e. its log. The recovery factor stays a user input.
- Analog ratios are lifetime averages; an oil well's late-life gas-oil ratio rises
  as reservoir pressure falls and would overstate a new well's gas.

**Decline curves** (`production/decline.py`)
- Fitted to the main product: oil rate (bbl/day) for oil wells, gas rate (Mcf/day)
  for gas wells. The other stream is forecast as a ratio to it.
- Fits start at peak rate by default; the fit window can be moved later to skip a
  workover or refrac.
- Zero and unreported months are excluded. Fits use log(rate) with a robust
  (soft L1) loss, so partial-downtime months don't drag the curve down.
- Hyperbolic forecasts switch to exponential at a terminal decline (default 8%/yr),
  so b > 1 can't forecast infinite reserves.
- The dashboard flags fits where b hits its 2.0 limit or the months scatter widely:
  long histories often contain refracs that one curve can't describe.

**Economics** (`production/economics.py`)
- Monthly cash flow: net revenue (after royalty) − severance and ad valorem taxes −
  fixed and variable LOE ($/bbl and $/Mcf) − capex, discounted mid-month.
- Production stops at the economic limit (last month with positive operating cash).
- Past months use historical WTI and Henry Hub; forecast months use the price deck.
  After the deck's last year, prices stay flat.
- Default assumptions (25% royalty, Texas severance 4.6% oil / 7.5% gas, LOE and
  capex by well type) are typical values, all editable in the dashboard.

**Cross-check** (`production/appraisal.py`)
- Each well type has a typical and a plausible recovery factor range, e.g. tight
  oil 3–15% (plausible 1–30%), conventional gas 50–85% (plausible 30–95%).
  Outside the plausible range, the log and the production almost certainly don't
  describe the same rock volume.

## Limitations

- RRC oil volumes are per lease; splitting them equally between wellbores is an
  approximation (wells on a lease rarely produce equally).
- The API lookup covers Texas (state code 42) and Kansas oil wells (15). Kansas
  wells must have a LAS file in the KGS index, and Kansas data is fetched live, not
  stored: KGS licenses the monthly volumes for research use, not redistribution.
- Spreading Kansas sales evenly within each year smooths out real month-to-month
  changes.
- The price deck is entered by hand (flat, escalating or custom); there is no
  live futures strip.
- Gas in place uses a single Bg; it doesn't model adsorbed gas, which matters in
  some shales.
