# PETE Production Analyzer

Oil & gas production analysis and valuation: pulls public production data from the
Texas Railroad Commission, fits Arps decline curves to forecast EUR, and runs a
discounted cash flow model to value the asset. Connects to the
[PETE Python Log Reader](../PETE%20Python%20Log%20Reader) to build a full upstream
appraisal dashboard: logs → volumetrics → decline curves → valuation.

Status: project setup. No analysis code yet.

## Setup (Ubuntu / WSL)

Uses the same virtual environment as the log reader. Run from this folder:

```bash
source ~/.venvs/well-log/bin/activate
pip install -r requirements.txt
```

This also installs the log reader as an editable package, so `import welllog` works
and picks up changes to the log reader immediately.
