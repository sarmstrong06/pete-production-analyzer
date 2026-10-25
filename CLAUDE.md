# PETE Production Analyzer

Oil & gas production data analyzer and valuation dashboard.

## Goal

1. Pull public production data (Texas Railroad Commission) for specific wells, leases, or basins.
2. Fit Arps decline curves (exponential, hyperbolic, harmonic) and forecast estimated ultimate recovery (EUR).
3. Apply a discounted cash flow (DCF) model against historical and forward oil/gas prices to judge whether an asset or energy equity looks undervalued.
4. Present results in an interactive dashboard.

## Tech stack

Python, Pandas, NumPy, SciPy (curve fitting), Matplotlib/Plotly.

## Domain notes

- Decline forecasting from public data uses Arps equations. Darcy's Law needs reservoir properties (permeability, pressure, viscosity) that public production data doesn't include, so it's background theory here, not a forecasting method.
- BOPD (barrels of oil per day) is the production rate that feeds the DCF: revenue = rate × price, minus royalties, operating costs, and severance taxes, then discounted.
- RRC reports oil production by lease, not by individual well. Gas wells are reported individually. Account for this when allocating volumes.

## Integration with the log reader (upstream appraisal dashboard)

End goal: one dashboard covering the appraisal workflow, from rock to dollars:

1. **Logs** (`..\PETE Python Log Reader`, package `welllog`): VSH, PHI, SW, net pay
2. **Volumetrics**: OOIP = 7758 × A × h × φ × (1 − Sw) / Bo (STB)
3. **Production** (this project): Arps fit and EUR
4. **Cross-check**: recovery factor = EUR / OOIP (flag values outside a plausible range)
5. **Valuation** (this project): DCF, NPV, breakeven price

Architecture decisions:

- Keep two separate repos. The log reader stays the source of truth for petrophysics; don't copy its functions here.
- Install the log reader into this project's virtual environment as an editable package (`pip install -e "..\PETE Python Log Reader"`) and `import welllog`. This requires the log reader to have a `pyproject.toml`.
- The dashboard lives in this repo, because it is the downstream consumer of both.
- Join wells by **API number** (14-digit, normalized to digits only). LAS headers carry it in `API` or `UWI`, and RRC data uses API numbers. Drainage area (A) and Bo aren't in either dataset, so they are user inputs on the dashboard.

The log reader provides `welllog.petrophysics.pay_flag` (cutoffs: VSH ≤ 0.4, PHI ≥ 0.08, SW ≤ 0.5; missing data never counts as pay) and `welllog.volumetrics.net_pay_summary` / `ooip`. Use these rather than reimplementing them.
