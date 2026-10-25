"""Upstream appraisal dashboard: well log -> oil in place -> decline curve -> value.

Run from the project folder with:  streamlit run app.py
"""

from dataclasses import replace
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import welllog
from welllog.petrophysics import archie_sw, density_porosity, pay_flag, shale_volume
from welllog.volumetrics import net_pay_summary, ooip

from production.appraisal import full_volumes, priced_wti, recovery_factor_check
from production.decline import (MODELS, eur, fit_arps, forecast,
                                nominal_from_effective_annual, prepare_history)
from production.economics import (Assumptions, breakeven_oil_price, cash_flow,
                                  load_wti, summarize)
from production.rrc import load_production

ROOT = Path(__file__).parent
SAMPLE_LAS = Path(welllog.__file__).resolve().parent.parent / "data" / "sample.las"

# Categorical slots in fixed order (validated palette); grays for chrome
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED, GRID = "#898781", "rgba(137,135,129,0.25)"  # recessive grid in light and dark themes
MODEL_COLOR = {"hyperbolic": ORANGE, "exponential": AQUA, "harmonic": YELLOW}

# Common LAS mnemonics for each curve, first match wins
ALIASES = {
    "GR": ["GR", "GRC", "SGR", "CGR", "GR_EDTC"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "ZDEN"],
    "RT": ["RT", "ILD", "RILD", "LLD", "AT90", "RDEP", "RD"],
}

st.set_page_config(page_title="Upstream Appraisal", page_icon="🛢️", layout="wide")


def style(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.12, yanchor="top", x=0),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    # Log axes: one gridline and label per decade, not the default minor clutter
    fig.for_each_xaxis(lambda ax: ax.update(dtick=1) if ax.type == "log" else None)
    fig.for_each_yaxis(lambda ax: ax.update(dtick=1) if ax.type == "log" else None)
    return fig


def money(x):
    if x is None or np.isnan(x):
        return "n/a"
    return f"-${abs(x)/1e6:,.2f}M" if x < 0 else f"${x/1e6:,.2f}M"


@st.cache_data
def load_data():
    prod = load_production(ROOT / "data" / "spraberry_leases.csv")
    wti = load_wti(ROOT / "data" / "wti_monthly.csv")
    return prod, wti


@st.cache_data
def read_las(text):
    las = lasio.read(text)
    df = las.df().reset_index()
    df = df.rename(columns={df.columns[0]: "DEPT"})
    header = {item.mnemonic: str(item.value) for item in las.well}
    params = {item.mnemonic: item.value for item in las.params}
    units = {c.mnemonic: c.unit for c in las.curves}
    return df, header, params, units


def pick_curve(label, key, columns):
    default = next((c for c in ALIASES[key] if c in columns), columns[0])
    return st.selectbox(label, columns, index=columns.index(default), key=f"curve_{key}")


prod, wti = load_data()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Inputs")
    leases = (prod.groupby(["lease_no", "lease_name"]).size().reset_index()
              .sort_values("lease_name"))
    labels = [f"{r.lease_name} ({r.lease_no})" for r in leases.itertuples()]
    choice = st.selectbox("Production lease (RRC)", labels)
    lease_no = int(leases.iloc[labels.index(choice)]["lease_no"])
    lease = prod[prod["lease_no"] == lease_no]
    st.caption(f"Spraberry (Trend Area), District 08 · Operator: "
               f"{lease['operator'].dropna().iloc[-1]} · {len(lease)} months reported")

    st.divider()
    source = st.radio("Well log (LAS)", ["Bundled sample", "Upload a LAS file"])
    if source == "Upload a LAS file":
        up = st.file_uploader("LAS file", type=["las", "LAS"])
        las_text = up.getvalue().decode("utf-8", errors="ignore") if up else None
    else:
        las_text = SAMPLE_LAS.read_text()
    if las_text is None:
        st.info("Upload a LAS file, or switch back to the bundled sample.")
        st.stop()

    log, header, las_params, units = read_las(las_text)
    api = header.get("API") or header.get("UWI") or ""
    st.caption(f"Log: {header.get('WELL', 'unnamed well')}" + (f" · API {api}" if api.strip() else ""))
    st.caption("The RRC lease data has no API numbers (oil is reported by lease), "
               "so the log and the lease are paired here by hand. Pair a log with "
               "the lease it was run in for a meaningful cross-check.")

summary_tab, log_tab, vol_tab, dca_tab, econ_tab = st.tabs(
    ["Summary", "1 · Well log", "2 · Volumetrics", "3 · Decline curve", "4 · Economics"])

# ---------------------------------------------------------------- 1. well log
with log_tab:
    with st.expander("Interpretation parameters", expanded=False):
        cols = [c for c in log.columns if c != "DEPT"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Curves**")
            gr_col = pick_curve("Gamma ray", "GR", cols)
            rhob_col = pick_curve("Bulk density", "RHOB", cols)
            rt_col = pick_curve("Deep resistivity", "RT", cols)
        with c2:
            st.markdown("**Rock and fluid**")
            gr_clean = st.number_input("GR clean sand (API)", value=25.0)
            gr_shale = st.number_input("GR shale (API)", value=110.0)
            rho_matrix = st.number_input("Matrix density (g/cc)", value=2.65,
                                         help="2.65 sandstone, 2.71 limestone, 2.87 dolomite")
            rho_fluid = st.number_input("Fluid density (g/cc)", value=1.0)
            rw_default = float(las_params.get("RW", 0.05) or 0.05)
            rw = st.number_input("Rw, formation water resistivity (ohm-m)", value=rw_default,
                                 format="%.3f", help="Read from the LAS ~Parameter section when present")
        with c3:
            st.markdown("**Pay cutoffs**")
            vsh_max = st.number_input("Shale volume ≤", value=0.40, step=0.05)
            phi_min = st.number_input("Porosity ≥", value=0.08, step=0.01)
            sw_max = st.number_input("Water saturation ≤", value=0.50, step=0.05)

    log["VSH"] = shale_volume(log[gr_col], gr_clean, gr_shale)
    log["PHI"] = density_porosity(log[rhob_col], rho_matrix, rho_fluid)
    log["SW"] = archie_sw(log[rt_col], log["PHI"], rw)
    log["PAY"] = pay_flag(log["VSH"], log["PHI"], log["SW"], vsh_max, phi_min, sw_max)
    pay = net_pay_summary(log["DEPT"], log["PAY"], log["PHI"], log["SW"])
    missing = log[log[["VSH", "PHI", "SW"]].isna().any(axis=1)]

    m = st.columns(4)
    m[0].metric("Net pay", f"{pay['net_pay_ft']:.1f} ft")
    m[1].metric("Net-to-gross", f"{pay['net_to_gross']:.0%}")
    m[2].metric("Avg porosity (pay)", f"{pay['avg_phi']:.1%}" if pay["net_pay_ft"] else "n/a")
    m[3].metric("Avg water saturation (pay)", f"{pay['avg_sw']:.1%}" if pay["net_pay_ft"] else "n/a")
    if not missing.empty:
        st.warning(f"Missing data at {len(missing)} depth(s), not counted as pay: "
                   + ", ".join(f"{d:g} ft" for d in missing["DEPT"]), icon="⚠️")

    tracks = [(gr_col, "GR (API)", BLUE, False), (rt_col, "Rt (ohm-m)", BLUE, True),
              ("PHI", "Porosity", BLUE, False), ("SW", "Water sat.", BLUE, False)]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[t[1] for t in tracks])
    step = float(np.median(np.diff(log["DEPT"])))
    # Shade pay intervals across every track
    for i, (col, name, color, logx) in enumerate(tracks, start=1):
        fig.add_trace(go.Scatter(x=log[col], y=log["DEPT"], mode="lines", name=name,
                                 line=dict(color=color, width=2), showlegend=False,
                                 hovertemplate="%{x:.3g} at %{y:g} ft<extra></extra>"),
                      row=1, col=i)
        if logx:
            fig.update_xaxes(type="log", row=1, col=i)
    # Shade pay intervals across every track (after the traces: plotly skips empty subplots)
    runs = (log["PAY"] != log["PAY"].shift()).cumsum()
    for _, run in log[log["PAY"]].groupby(runs[log["PAY"]]):
        for i in range(1, 5):
            fig.add_hrect(y0=run["DEPT"].min() - step / 2, y1=run["DEPT"].max() + step / 2,
                          fillcolor=AQUA, opacity=0.25, line_width=0, layer="below", row=1, col=i)
    fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=1, col=1)
    style(fig, height=560).update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded green: pay, where all three cutoffs pass.")
    with st.expander("Pay zone table"):
        st.dataframe(log.loc[log["PAY"], ["DEPT", "VSH", "PHI", "SW"]].round(3), hide_index=True)

# ---------------------------------------------------------------- 2. volumetrics
with vol_tab:
    st.markdown("Original oil in place: **OOIP = 7758 × A × h × φ × (1 − Sw) / Bo**")
    c1, c2 = st.columns(2)
    area = c1.number_input("Drainage area, A (acres)", value=200.0, step=10.0,
                           help="Neither dataset contains this. A Spraberry horizontal well with a "
                                "~10,000 ft lateral and ~880 ft well spacing drains about 200 acres.")
    bo = c2.number_input("Oil formation volume factor, Bo (rb/stb)", value=1.30, step=0.05,
                         help="Reservoir barrels per stock tank barrel. Oil shrinks as gas leaves it at the surface.")
    oil_in_place = (ooip(area, pay["net_pay_ft"], pay["avg_phi"], pay["avg_sw"], bo)
                    if pay["net_pay_ft"] else 0.0)
    m = st.columns(4)
    m[0].metric("OOIP", f"{oil_in_place/1e6:,.2f} MMSTB" if oil_in_place >= 1e6 else f"{oil_in_place:,.0f} STB")
    m[1].metric("Net pay, h", f"{pay['net_pay_ft']:.1f} ft")
    m[2].metric("Porosity, φ", f"{pay['avg_phi']:.1%}" if pay["net_pay_ft"] else "n/a")
    m[3].metric("Water saturation, Sw", f"{pay['avg_sw']:.1%}" if pay["net_pay_ft"] else "n/a")
    st.caption("h, φ and Sw come from the well log tab. Sw is averaged by pore volume.")

# ---------------------------------------------------------------- 3. decline curve
with dca_tab:
    producing = lease[lease["bopd"] > 0]
    peak_date = producing.loc[producing["bopd"].idxmax(), "date"]
    dates = list(producing.loc[producing["date"] >= peak_date, "date"])

    c1, c2, c3, c4 = st.columns(4)
    start = c1.selectbox("Fit from", dates, format_func=lambda d: d.strftime("%b %Y"),
                         help="Defaults to peak month. Move it later to fit only the decline "
                              "after a workover or refrac.")
    model = c2.radio("Forecast model", MODELS, index=MODELS.index("hyperbolic"), horizontal=False)
    d_term = c3.number_input("Terminal decline (%/yr)", value=8.0, step=1.0,
                             help="Hyperbolic decline switches to exponential at this rate, "
                                  "so b > 1 can't forecast infinite reserves.") / 100
    q_limit = c4.number_input("Forecast cutoff (bbl/day)", value=5.0, step=1.0,
                              help="Forecast stops below this rate. The economics tab applies "
                                   "its own economic limit on top.")

    hist = prepare_history(lease, start)
    fits = {mdl: fit_arps(hist, mdl) for mdl in MODELS}
    forecasts = {mdl: forecast(f, hist, q_limit=q_limit, d_term_annual=d_term) for mdl, f in fits.items()}
    fit, fc = fits[model], forecasts[model]
    lease_eur = eur(lease, fc)
    cum = float(lease["oil_bbl"].sum())

    m = st.columns(4)
    m[0].metric("EUR", f"{lease_eur/1e3:,.0f} Mbbl")
    m[1].metric("Produced to date", f"{cum/1e3:,.0f} Mbbl")
    m[2].metric("Remaining (forecast)", f"{(lease_eur - cum)/1e3:,.0f} Mbbl")
    m[3].metric("Initial decline / b", f"{fit.di_annual_eff:.0%}/yr · b={fit.b:.2f}")

    fig = go.Figure()
    excluded = lease[(lease["date"] < hist["date"].iloc[0]) | ~(lease["bopd"] > 0)]
    fig.add_trace(go.Scatter(x=excluded["date"], y=excluded["bopd"].clip(lower=0.5), mode="markers",
                             name="Not fitted", marker=dict(color=MUTED, size=8, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["bopd"], mode="markers", name="Actual (fitted)",
                             marker=dict(color=BLUE, size=8, line=dict(color="white", width=1))))
    for mdl in MODELS:
        f = fits[mdl]
        t_all = np.arange(0, int(forecasts[mdl]["t"].max() if len(forecasts[mdl]) else hist["t"].max()) + 1)
        d_all = [f.t0 + pd.DateOffset(months=int(t)) for t in t_all]
        q_all = f.rate(t_all, nominal_from_effective_annual(d_term))
        fig.add_trace(go.Scatter(x=d_all, y=q_all, mode="lines", name=f"{mdl.capitalize()}",
                                 line=dict(color=MODEL_COLOR[mdl], width=3 if mdl == model else 2,
                                           dash=None if mdl == model else "dot")))
    fig.add_vline(x=lease["date"].max(), line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(type="log", title_text="Oil rate (bbl/day, log scale)")
    style(fig, height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed vertical line: last reported month. Solid curve: selected forecast model.")

    table = pd.DataFrame([{
        "Model": mdl.capitalize(), "qi (bbl/d)": round(f.qi), "Di (%/yr)": f"{f.di_annual_eff:.0%}",
        "b": round(f.b, 2), "Fit error (log RMSE)": round(f.rmse_log, 2),
        "EUR (Mbbl)": round(eur(lease, forecasts[mdl]) / 1e3)} for mdl, f in fits.items()])
    st.dataframe(table, hide_index=True)
    st.caption("Fit uses a robust loss that down-weights months with partial downtime. "
               "Zero and unreported months are left out.")

# ---------------------------------------------------------------- 4. economics
with econ_tab:
    recent_wti = float(wti["wti"].tail(12).mean())
    with st.expander("Price and cost assumptions", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        a = Assumptions(
            oil_price=c1.number_input("WTI oil price, forecast ($/bbl)", value=round(recent_wti, 2),
                                      help="Defaults to the last 12 months' average WTI (FRED). "
                                           "Past months use actual historical WTI."),
            oil_diff=c1.number_input("Oil differential ($/bbl)", value=-1.50, step=0.25),
            gas_price=c1.number_input("Henry Hub gas ($/MCF)", value=2.50, step=0.25),
            gas_diff_pct=c1.number_input("Gas realized (% of Henry Hub)", value=70.0, step=5.0) / 100,
            working_interest=c2.number_input("Working interest (%)", value=100.0, step=5.0) / 100,
            royalty=c2.number_input("Royalty (%)", value=25.0, step=1.0) / 100,
            severance_oil=c2.number_input("Oil severance tax (%)", value=4.6, step=0.1) / 100,
            severance_gas=c2.number_input("Gas severance tax (%)", value=7.5, step=0.1) / 100,
            ad_valorem=c3.number_input("Ad valorem tax (%)", value=2.0, step=0.5) / 100,
            loe_fixed=c3.number_input("Fixed LOE ($/month)", value=8000.0, step=500.0),
            loe_var=c3.number_input("Variable LOE ($/bbl)", value=4.0, step=0.5),
            capex=c4.number_input("Drill & complete capex ($)", value=8_000_000.0, step=250_000.0),
            discount_rate=c4.number_input("Discount rate (%/yr)", value=10.0, step=1.0) / 100,
        )

    vol = full_volumes(lease, fc)
    full_cf = cash_flow(vol, a, oil_prices=priced_wti(vol, wti, a.oil_price), include_capex=True)
    full = summarize(full_cf)
    pdp_vol = vol[~vol["actual"]]
    pdp_cf = cash_flow(pdp_vol, a)
    pdp = summarize(pdp_cf)
    breakeven = breakeven_oil_price(vol, a)
    rate_pct = f"{a.discount_rate:.0%}"

    st.subheader("Full-cycle: was drilling this well worth it?")
    st.caption("From first production, including capex. Months already produced use actual "
               "historical WTI; forecast months use the flat price above.")
    m = st.columns(4)
    m[0].metric(f"NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    m[2].metric("Payout", f"{full.payout_months:.0f} mo" if not np.isnan(full.payout_months) else "never")
    m[3].metric("Breakeven WTI ($/bbl)", f"${breakeven:,.2f}" if not np.isnan(breakeven) else "> $300")

    st.subheader("Remaining value: what is the producing well worth today?")
    st.caption("Forecast only (proved developed producing, PDP), at the flat price, no capex.")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"PDP value @ {rate_pct}", money(pdp.npv))
    c2.metric("Remaining economic oil", f"{pdp.net_oil/1e3:,.0f} Mbbl")
    asking = c3.number_input("Asking price for this well ($)", value=0.0, step=100_000.0,
                             help="Enter a price to compare against the PDP value.")
    if asking > 0:
        gap = (pdp.npv - asking) / asking
        if gap > 0.1:
            st.success(f"Undervalued: PDP value is {gap:.0%} above the asking price.", icon="✅")
        elif gap < -0.1:
            st.error(f"Overvalued: PDP value is {-gap:.0%} below the asking price.", icon="⛔")
        else:
            st.info("Fairly priced: PDP value is within 10% of the asking price.", icon="ℹ️")

    c1, c2 = st.columns(2)
    annual = full_cf.assign(year=full_cf["date"].dt.year).groupby("year")["net_cf"].sum().reset_index()
    fig = go.Figure(go.Bar(x=annual["year"], y=annual["net_cf"] / 1e6, marker_color=BLUE,
                           name="Net cash flow", hovertemplate="%{x}: $%{y:.2f}M<extra></extra>"))
    fig.update_yaxes(title_text="Net cash flow ($M)")
    style(fig, height=380).update_layout(title="Annual net cash flow (full-cycle)", hovermode="closest")
    c1.plotly_chart(fig, width="stretch")

    prices = np.arange(30, 125, 5)
    sens_full = [cash_flow(vol, replace(a, oil_price=p), oil_prices=priced_wti(vol, wti, p),
                           include_capex=True)["pv"].sum() / 1e6 for p in prices]
    sens_pdp = [cash_flow(pdp_vol, replace(a, oil_price=p))["pv"].sum() / 1e6 for p in prices]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=sens_full, mode="lines+markers", name="Full-cycle NPV",
                             line=dict(color=BLUE, width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=prices, y=sens_pdp, mode="lines+markers", name="PDP value",
                             line=dict(color=ORANGE, width=2), marker=dict(size=8)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="Forecast WTI ($/bbl)")
    fig.update_yaxes(title_text=f"Value @ {rate_pct} ($M)")
    style(fig, height=380).update_layout(title="Oil price sensitivity")
    c2.plotly_chart(fig, width="stretch")

    with st.expander("Monthly cash flow table (full-cycle)"):
        st.dataframe(full_cf.assign(date=full_cf["date"].dt.strftime("%Y-%m")).round(2), hide_index=True)

# ---------------------------------------------------------------- summary
with summary_tab:
    st.title("Upstream appraisal")
    st.caption(f"Log: {header.get('WELL', 'unnamed')} · Lease: {choice} · "
               f"Forecast model: {model} · WTI ${a.oil_price:,.2f}/bbl")

    st.markdown("#### Rock → oil in place → recovery → dollars")
    m = st.columns(4)
    m[0].metric("Net pay (log)", f"{pay['net_pay_ft']:.1f} ft")
    m[1].metric("Oil in place",
                f"{oil_in_place/1e6:,.2f} MMSTB" if oil_in_place >= 1e6 else f"{oil_in_place/1e3:,.0f} Mbbl")
    m[2].metric("EUR", f"{lease_eur/1e3:,.0f} Mbbl")
    rf, verdict = recovery_factor_check(lease_eur, oil_in_place)
    m[3].metric("Recovery factor", f"{rf:.1%}" if not np.isnan(rf) else "n/a")

    if np.isnan(rf) or rf > 0.30 or rf < 0.01:
        st.error(f"**Cross-check failed.** {verdict}", icon="⛔")
    elif 0.03 <= rf <= 0.15:
        st.success(f"**Cross-check passed.** {verdict}", icon="✅")
    else:
        st.warning(f"**Cross-check: review.** {verdict}", icon="⚠️")

    m = st.columns(4)
    m[0].metric(f"Full-cycle NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    m[2].metric(f"PDP value @ {rate_pct}", money(pdp.npv))
    m[3].metric("Breakeven WTI ($/bbl)", f"${breakeven:,.2f}" if not np.isnan(breakeven) else "> $300")

    if source == "Bundled sample":
        st.info("The bundled log is a 12 ft synthetic test well, not a log from this lease. "
                "Its oil in place is far too small for a horizontal Spraberry well, which is "
                "why the cross-check fails. Upload a real LAS from the same lease for a "
                "meaningful result.", icon="ℹ️")
