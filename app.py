"""Upstream appraisal dashboard: well log -> oil in place -> production forecast -> value.

The LAS log is the primary input. Production data is optional:
- Without it, EUR = oil in place × an assumed recovery factor, spread over time
  with an analog type curve (how a well is appraised before it's drilled).
- With it (an example lease or any Texas lease fetched from the RRC), the
  forecast comes from the well's actual decline, and the recovery factor
  becomes a cross-check between the log and the production.

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

from production.appraisal import (forecast_volumes, full_volumes, priced_wti,
                                  producing_gor, recovery_factor_check)
from production.decline import (MODELS, eur, fit_arps, forecast,
                                nominal_from_effective_annual, prepare_history, type_curve)
from production.economics import (Assumptions, breakeven_oil_price, cash_flow,
                                  load_wti, summarize)
from production.rrc import DISTRICT_CODES, clean_production, fetch_lease, load_production

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

# Starting assumptions by well type; every one is editable in the dashboard
WELL_TYPES = {
    "Vertical (conventional)": dict(
        area=40.0, rf=15.0, b=0.6, d_eff=35.0, q_limit=3.0, gor=1000.0,
        capex=2_500_000.0, loe_fixed=4000.0, loe_var=6.0,
        note="A vertical well in a conventional sandstone drains roughly 40 acres "
             "and recovers 10-25% of oil in place on primary production."),
    "Horizontal (tight oil)": dict(
        area=200.0, rf=8.0, b=1.1, d_eff=70.0, q_limit=5.0, gor=2000.0,
        capex=8_000_000.0, loe_fixed=8000.0, loe_var=4.0,
        note="A fractured horizontal well with a ~10,000 ft lateral drains roughly "
             "200 acres and recovers 3-15% of oil in place."),
}

# Appraisal modes (sidebar step 3) and where a lease comes from (step 4)
TYPICAL = "Prospect: typical assumptions"
ANALOG = "Prospect: decline from an analog well"
OWN = "Producing well: its own production"
EXAMPLE, FETCH = "Example Spraberry lease", "Fetch any Texas lease"
MODE_HELP = {
    TYPICAL: "No production data. EUR = oil in place × recovery factor, with a typical "
             "decline shape for the well type.",
    ANALOG: "A similar nearby well's production sets the decline shape and gas-oil ratio. "
            "Your log still sets the size: oil in place × recovery factor.",
    OWN: "This well's own production history sets the forecast. The recovery factor "
         "(EUR ÷ oil in place) becomes a cross-check between the log and the production.",
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


def barrels(x):
    return f"{x/1e6:,.2f} MMbbl" if x >= 1e6 else f"{x/1e3:,.0f} Mbbl"


@st.cache_data
def load_data():
    prod = load_production(ROOT / "data" / "spraberry_leases.csv")
    wti = load_wti(ROOT / "data" / "wti_monthly.csv")
    return prod, wti


@st.cache_data(ttl=86400, show_spinner="Fetching production from the Railroad Commission...")
def fetch_cached(lease_no, district):
    return clean_production(fetch_lease(lease_no, district))


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
    st.header("1 · Well log")
    source = st.radio("LAS file", ["Bundled sample", "Upload a LAS file"], label_visibility="collapsed")
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

    st.header("2 · Well type")
    well_type = st.radio("Well type", list(WELL_TYPES), label_visibility="collapsed")
    wt = WELL_TYPES[well_type]
    st.caption(wt["note"] + " Sets starting assumptions; all are editable.")

    st.header("3 · Appraisal mode")
    mode = st.radio("Appraisal mode", [TYPICAL, ANALOG, OWN], label_visibility="collapsed")
    st.caption(MODE_HELP[mode])

    lease, lease_label = None, ""
    if mode != TYPICAL:
        st.header("4 · " + ("Analog lease" if mode == ANALOG else "This well's lease"))
        prod_source = st.radio("Lease source", [EXAMPLE, FETCH], label_visibility="collapsed")
    else:
        prod_source = None
    if prod_source == EXAMPLE:
        leases = prod.groupby(["lease_no", "lease_name"]).size().reset_index().sort_values("lease_name")
        labels = [f"{r.lease_name} ({r.lease_no})" for r in leases.itertuples()]
        choice = st.selectbox("Lease", labels)
        lease = prod[prod["lease_no"] == int(leases.iloc[labels.index(choice)]["lease_no"])]
    elif prod_source == FETCH:
        with st.form("fetch"):
            c1, c2 = st.columns([3, 2])
            lease_in = c1.text_input("Oil lease number", placeholder="e.g. 50678")
            district = c2.selectbox("District", list(DISTRICT_CODES), index=list(DISTRICT_CODES).index("08"))
            submitted = st.form_submit_button("Fetch from RRC", width="stretch")
        if submitted:
            if lease_in.strip().isdigit():
                st.session_state["fetched_lease"] = (int(lease_in.strip()), district)
            else:
                st.error("Lease number must be digits only.")
        if "fetched_lease" in st.session_state:
            try:
                lease = fetch_cached(*st.session_state["fetched_lease"])
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # network trouble, site changes
                st.error(f"Couldn't reach the Railroad Commission: {e}")
        st.caption("The lease number is on the lease sign, drilling permits, or the RRC's "
                   "public GIS map. Oil leases only; gas wells are reported separately.")

    if lease is not None:
        producing_months = int((lease["bopd"] > 0).sum())
        lease_label = f"{lease['lease_name'].iloc[0]} ({lease['lease_no'].iloc[0]})"
        op = lease["operator"].dropna()
        st.caption(f"{lease_label} · {lease['field'].iloc[0]} · "
                   f"{op.iloc[-1] if len(op) else 'operator n/a'} · {producing_months} producing months")
        if producing_months < 6:
            st.warning("Fewer than 6 producing months: too little history to fit a decline. "
                       "Using typical assumptions instead.", icon="⚠️")
            lease = None
        elif mode == OWN:
            st.caption("PDQ lease data has no API numbers, so make sure this lease is the well "
                       "the log was run in; otherwise the cross-check is meaningless.")

has_prod = mode == OWN and lease is not None      # forecast from this well's own history
use_analog = mode == ANALOG and lease is not None  # decline shape borrowed from another well

summary_tab, log_tab, vol_tab, fc_tab, econ_tab = st.tabs(
    ["Summary", "1 · Well log", "2 · Volumetrics", "3 · Production forecast", "4 · Economics"])

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
    area = c1.number_input("Drainage area, A (acres)", value=wt["area"], step=10.0, key=f"area_{well_type}",
                           help="The area one well drains. Not in the log; set by well type and spacing.")
    bo = c2.number_input("Oil formation volume factor, Bo (rb/stb)", value=1.30, step=0.05,
                         help="Reservoir barrels per stock tank barrel. Oil shrinks as gas leaves it at the surface.")
    oil_in_place = (ooip(area, pay["net_pay_ft"], pay["avg_phi"], pay["avg_sw"], bo)
                    if pay["net_pay_ft"] else 0.0)
    m = st.columns(4)
    m[0].metric("OOIP", barrels(oil_in_place))
    m[1].metric("Net pay, h", f"{pay['net_pay_ft']:.1f} ft")
    m[2].metric("Porosity, φ", f"{pay['avg_phi']:.1%}" if pay["net_pay_ft"] else "n/a")
    m[3].metric("Water saturation, Sw", f"{pay['avg_sw']:.1%}" if pay["net_pay_ft"] else "n/a")
    st.caption("h, φ and Sw come from the well log tab. Sw is averaged by pore volume.")

# ---------------------------------------------------------------- 3. production forecast
with fc_tab:
    if has_prod:
        st.markdown(f"**Decline curve fitted to actual production** · {lease_label}")
        producing = lease[lease["bopd"] > 0]
        peak_date = producing.loc[producing["bopd"].idxmax(), "date"]
        dates = list(producing.loc[producing["date"] >= peak_date, "date"])

        c1, c2, c3, c4 = st.columns(4)
        start = c1.selectbox("Fit from", dates, format_func=lambda d: d.strftime("%b %Y"),
                             help="Defaults to peak month. Move it later to fit only the decline "
                                  "after a workover or refrac.")
        model = c2.radio("Forecast model", MODELS, index=MODELS.index("hyperbolic"))
        d_term = c3.number_input("Terminal decline (%/yr)", value=8.0, step=1.0,
                                 help="Hyperbolic decline switches to exponential at this rate, "
                                      "so b > 1 can't forecast infinite reserves.") / 100
        q_limit = c4.number_input("Forecast cutoff (bbl/day)", value=5.0, step=1.0,
                                  help="Forecast stops below this rate. The economics tab applies "
                                       "its own economic limit on top.")

        hist = prepare_history(lease, start)
        if len(hist) < 4:
            st.error("Fewer than 4 producing months after the chosen start; pick an earlier month.")
            st.stop()
        fits = {mdl: fit_arps(hist, mdl) for mdl in MODELS}
        forecasts = {mdl: forecast(f, hist, q_limit=q_limit, d_term_annual=d_term) for mdl, f in fits.items()}
        fit, fc = fits[model], forecasts[model]
        well_eur = eur(lease, fc)
        cum = float(lease["oil_bbl"].sum())

        m = st.columns(4)
        m[0].metric("EUR", barrels(well_eur))
        m[1].metric("Produced to date", barrels(cum))
        m[2].metric("Remaining (forecast)", barrels(well_eur - cum))
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
    else:
        model = "type curve"
        b_default, deff_default, key_sfx = wt["b"], wt["d_eff"], well_type
        if use_analog:
            # Borrow the analog's decline shape: fit it, then use its b and first-year decline
            analog_hist = prepare_history(lease)
            analog_fit = fit_arps(analog_hist, "hyperbolic")
            analog_fc = forecast(analog_fit, analog_hist)
            b_default = float(np.clip(round(analog_fit.b, 2), 0.0, 2.0))
            deff_default = float(np.clip(round(analog_fit.di_annual_eff * 100), 1, 99))
            key_sfx = f"{well_type}_{lease['lease_no'].iloc[0]}"  # new analog -> fresh defaults
            st.markdown(f"**Forecast from the log, shaped like analog {lease_label}**: "
                        "your oil in place × recovery factor sets how much; the analog's actual "
                        "decline sets how fast.")
            m = st.columns(4)
            m[0].metric("Analog EUR", barrels(eur(lease, analog_fc)))
            m[1].metric("Analog peak rate", f"{analog_hist['bopd'].iloc[0]:,.0f} bbl/day")
            m[2].metric("Analog b-factor", f"{analog_fit.b:.2f}")
            m[3].metric("Analog first-year decline", f"{analog_fit.di_annual_eff:.0%}")
        else:
            st.markdown("**Forecast from the log** (no production data): EUR = oil in place × "
                        "recovery factor, spread over time with a typical decline shape for the well type.")
        c1, c2, c3 = st.columns(3)
        rf_assumed = c1.number_input("Recovery factor (%)", value=wt["rf"], step=1.0, key=f"rf_{well_type}",
                                     help="Share of oil in place the well will recover. The biggest "
                                          "uncertainty in a log-only appraisal. An analog can't supply "
                                          "this without its own log.") / 100
        first_prod = c1.date_input("First production", value=(pd.Timestamp.today() + pd.offsets.MonthBegin(1)).date())
        shape_src = "from the analog's fit" if use_analog else "typical for the well type"
        b = c2.number_input("b-factor", value=b_default, step=0.1, min_value=0.0, max_value=2.0,
                            key=f"b_{key_sfx}", help=f"Shape of the decline; {shape_src}.")
        d_eff = c2.number_input("First-year decline (%)", value=deff_default, step=5.0, min_value=1.0,
                                max_value=99.0, key=f"deff_{key_sfx}",
                                help=f"How much the rate drops in year one; {shape_src}.") / 100
        d_term = c3.number_input("Terminal decline (%/yr)", value=8.0, step=1.0) / 100
        q_limit = c3.number_input("Forecast cutoff (bbl/day)", value=wt["q_limit"], step=1.0,
                                  key=f"ql_{well_type}")

        well_eur = oil_in_place * rf_assumed
        try:
            fit, fc = type_curve(well_eur, pd.Timestamp(first_prod).replace(day=1), b, d_eff,
                                 d_term_annual=d_term, q_limit=q_limit)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        m = st.columns(4)
        m[0].metric("EUR", barrels(well_eur))
        m[1].metric("Initial rate", f"{fit.qi:,.0f} bbl/day" if len(fc) else "n/a")
        m[2].metric("First-year oil", barrels(float(fc["oil_bbl"].head(12).sum())))
        m[3].metric("Producing life", f"{len(fc)/12:.1f} years")
        if fc.empty:
            st.warning("Oil in place × recovery factor is too small to produce above the forecast "
                       "cutoff rate at all.", icon="⚠️")
        else:
            # Months on production, so a historical analog lines up with a future well
            fig = go.Figure()
            if use_analog:
                fig.add_trace(go.Scatter(x=analog_hist["t"], y=analog_hist["bopd"], mode="markers",
                                         name=f"Analog actual ({lease_label})",
                                         marker=dict(color=BLUE, size=8, line=dict(color="white", width=1)),
                                         hovertemplate="Month %{x}: %{y:,.0f} bbl/day<extra></extra>"))
                t_a = np.arange(0, int(max(analog_fc["t"].max() if len(analog_fc) else 0, analog_hist["t"].max())) + 1)
                fig.add_trace(go.Scatter(x=t_a, y=analog_fit.rate(t_a, nominal_from_effective_annual(d_term)),
                                         mode="lines", name="Analog fit", line=dict(color=BLUE, width=2, dash="dot"),
                                         hovertemplate="Month %{x}: %{y:,.0f} bbl/day<extra></extra>"))
            fig.add_trace(go.Scatter(x=fc["t"], y=fc["bopd"], mode="lines", name="Your well (forecast)",
                                     line=dict(color=ORANGE, width=3),
                                     hovertemplate="Month %{x}: %{y:,.1f} bbl/day<extra></extra>"))
            fig.update_xaxes(title_text="Months on production")
            fig.update_yaxes(type="log", title_text="Oil rate (bbl/day, log scale)")
            style(fig, height=440).update_layout(hovermode="closest", showlegend=use_analog)
            st.plotly_chart(fig, width="stretch")
            if use_analog:
                st.caption("Same shape, different size: your forecast follows the analog's decline, "
                           "scaled so it recovers your log's oil in place × recovery factor.")
        st.info("This forecast is only as good as its assumptions. If this well already produces, "
                "switch to 'Producing well' mode to use its actual decline.", icon="ℹ️")

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
            loe_fixed=c3.number_input("Fixed LOE ($/month)", value=wt["loe_fixed"], step=500.0,
                                      key=f"loef_{well_type}"),
            loe_var=c3.number_input("Variable LOE ($/bbl)", value=wt["loe_var"], step=0.5,
                                    key=f"loev_{well_type}"),
            capex=c4.number_input("Drill & complete capex ($)", value=wt["capex"], step=250_000.0,
                                  key=f"capex_{well_type}"),
            discount_rate=c4.number_input("Discount rate (%/yr)", value=10.0, step=1.0) / 100,
        )
        if not has_prod:
            # A new well's whole life is being forecast, so use the analog's lifetime GOR
            # (late-life GOR rises as reservoir pressure drops and would overstate gas)
            gor_default = round(producing_gor(lease, months=len(lease)) * 1000, -1) if use_analog else wt["gor"]
            gor = c4.number_input("Gas-oil ratio (scf/bbl)", value=float(gor_default), step=100.0,
                                  key=f"gor_{key_sfx}",
                                  help="Gas produced per barrel of oil"
                                       + (" (analog's lifetime average)." if use_analog else ".")) / 1000

    rate_pct = f"{a.discount_rate:.0%}"
    if has_prod:
        vol = full_volumes(lease, fc)
        oil_prices = priced_wti(vol, wti, a.oil_price)
        price_fn = lambda p: priced_wti(vol, wti, p)
    else:
        vol = forecast_volumes(fc, gor)
        if vol.empty:  # nothing to produce: the well still costs its capex
            vol = pd.DataFrame({"date": [pd.Timestamp(first_prod).replace(day=1)], "oil_bbl": [0.0],
                                "gas_mcf": [0.0], "actual": [False]})
        oil_prices = None
        price_fn = lambda p: None
    full_cf = cash_flow(vol, a, oil_prices=oil_prices, include_capex=True)
    full = summarize(full_cf)
    breakeven = breakeven_oil_price(vol, a)

    if has_prod:
        st.subheader("Full-cycle: was drilling this well worth it?")
        st.caption("From first production, including capex. Months already produced use actual "
                   "historical WTI; forecast months use the flat price above.")
    else:
        st.subheader("Drilling decision: is this well worth drilling?")
        st.caption("The log-based forecast at the flat price above, including capex.")
    m = st.columns(4)
    m[0].metric(f"NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    m[2].metric("Payout", f"{full.payout_months:.0f} mo" if not np.isnan(full.payout_months) else "never")
    m[3].metric("Breakeven WTI ($/bbl)", f"${breakeven:,.2f}" if not np.isnan(breakeven) else "> $300")

    pdp = None
    if has_prod:
        pdp_vol = vol[~vol["actual"]]
        pdp = summarize(cash_flow(pdp_vol, a))
        st.subheader("Remaining value: what is the producing well worth today?")
        st.caption("Forecast only (proved developed producing, PDP), at the flat price, no capex.")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"PDP value @ {rate_pct}", money(pdp.npv))
        c2.metric("Remaining economic oil", barrels(pdp.net_oil))
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, y=[cash_flow(vol, replace(a, oil_price=p), oil_prices=price_fn(p),
                               include_capex=True)["pv"].sum() / 1e6 for p in prices],
        mode="lines+markers", name="Full-cycle NPV", line=dict(color=BLUE, width=2), marker=dict(size=8)))
    if has_prod:
        fig.add_trace(go.Scatter(
            x=prices, y=[cash_flow(pdp_vol, replace(a, oil_price=p))["pv"].sum() / 1e6 for p in prices],
            mode="lines+markers", name="PDP value", line=dict(color=ORANGE, width=2), marker=dict(size=8)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="Forecast WTI ($/bbl)")
    fig.update_yaxes(title_text=f"Value @ {rate_pct} ($M)")
    style(fig, height=380).update_layout(title="Oil price sensitivity", showlegend=has_prod)
    c2.plotly_chart(fig, width="stretch")

    with st.expander("Monthly cash flow table (full-cycle)"):
        st.dataframe(full_cf.assign(date=full_cf["date"].dt.strftime("%Y-%m")).round(2), hide_index=True)

# ---------------------------------------------------------------- summary
with summary_tab:
    st.title("Upstream appraisal")
    st.caption(f"Log: {header.get('WELL', 'unnamed')} · {well_type} · "
               + (f"Producing well: {lease_label} · " if has_prod
                  else f"Prospect, decline from analog {lease_label} · " if use_analog
                  else "Prospect, typical assumptions · ") +
               f"WTI ${a.oil_price:,.2f}/bbl")

    st.markdown("#### Rock → oil in place → recovery → dollars")
    m = st.columns(4)
    m[0].metric("Net pay (log)", f"{pay['net_pay_ft']:.1f} ft")
    m[1].metric("Oil in place", barrels(oil_in_place))
    m[2].metric("EUR", barrels(well_eur))
    if has_prod:
        rf, verdict = recovery_factor_check(well_eur, oil_in_place)
        m[3].metric("Recovery factor", f"{rf:.1%}" if not np.isnan(rf) else "n/a")
        if np.isnan(rf) or rf > 0.30 or rf < 0.01:
            st.error(f"**Cross-check failed.** {verdict}", icon="⛔")
        elif 0.03 <= rf <= 0.15:
            st.success(f"**Cross-check passed.** {verdict}", icon="✅")
        else:
            st.warning(f"**Cross-check: review.** {verdict}", icon="⚠️")
    else:
        m[3].metric("Recovery factor (assumed)", f"{rf_assumed:.0%}")
        shape = f"the decline shape of analog {lease_label}" if use_analog else "a typical decline shape"
        st.info(f"EUR is oil in place × an assumed {rf_assumed:.0%} recovery factor, produced with "
                f"{shape}. Once the well has its own production, switch to 'Producing well' mode "
                "to test these assumptions against real performance.", icon="ℹ️")

    m = st.columns(4)
    m[0].metric(f"NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    if has_prod:
        m[2].metric(f"PDP value @ {rate_pct}", money(pdp.npv))
    else:
        m[2].metric("Payout", f"{full.payout_months:.0f} mo" if not np.isnan(full.payout_months) else "never")
    m[3].metric("Breakeven WTI ($/bbl)", f"${breakeven:,.2f}" if not np.isnan(breakeven) else "> $300")

    if source == "Bundled sample":
        st.info("The bundled log is a 12 ft synthetic test well with 3.5 ft of pay, so it holds "
                "very little oil. Upload a real LAS file to appraise a real well.", icon="ℹ️")
