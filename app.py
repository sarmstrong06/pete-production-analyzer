"""Upstream appraisal dashboard: well log -> volume in place -> production forecast -> value.

The LAS log is the primary input, for an oil or a gas well. Production data is
optional:
- Without it, EUR = volume in place × an assumed recovery factor, spread over
  time with a type curve (how a well is appraised before it's drilled).
- With an analog (a similar well), the analog's actual decline sets the shape.
- With the well's own production (found from the log's API number, an RRC
  lease number or gas well ID, or an example), the forecast comes from its
  actual decline, and the recovery factor becomes a cross-check between the
  log and the production.

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
from welllog.volumetrics import net_pay_summary, ogip, ooip

from production import kgs
from production.appraisal import (PRODUCTS, forecast_volumes, full_volumes, priced_history,
                                  recovery_factor_check, secondary_ratio)
from production.decline import (MODELS, eur, fit_arps, forecast,
                                nominal_from_effective_annual, prepare_history, type_curve)
from production.economics import (BREAKEVEN_MAX, Assumptions, breakeven_price, cash_flow,
                                  escalated_deck, load_henry_hub, load_wti, price_deck, summarize)
from production.rrc import (DISTRICT_CODES, clean_production, fetch_lease, lease_kind,
                            lease_wells, load_production, lookup_api, normalize_api)

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

# Starting assumptions by well type; every one is editable in the dashboard.
# ratio: gas-oil ratio (Mcf/bbl) for oil wells, condensate yield (bbl/Mcf) for gas wells.
# fvf: Bo (rb/stb) for oil, Bg (rcf/scf) for gas. q_limit: per day in the main product.
WELL_TYPES = {
    "Vertical oil (conventional)": dict(
        product="oil", area=40.0, rf=15.0, rf_typical=(0.10, 0.25), rf_plausible=(0.05, 0.40),
        label="a vertical conventional oil well", b=0.6, d_eff=35.0, q_limit=3.0, ratio=1.0,
        fvf=1.30, capex=2_500_000.0, loe_fixed=4000.0, loe_var=6.0, loe_var_gas=0.0,
        oil_diff=-1.50, gas_realized=70.0,
        note="A vertical well in a conventional sandstone or carbonate drains roughly 40 acres "
             "and recovers 10-25% of oil in place on primary production."),
    "Horizontal oil (tight)": dict(
        product="oil", area=200.0, rf=8.0, rf_typical=(0.03, 0.15), rf_plausible=(0.01, 0.30),
        label="a horizontal tight-oil well", b=1.1, d_eff=70.0, q_limit=5.0, ratio=2.0,
        fvf=1.30, capex=8_000_000.0, loe_fixed=8000.0, loe_var=4.0, loe_var_gas=0.0,
        oil_diff=-1.50, gas_realized=70.0,
        note="A fractured horizontal well with a ~10,000 ft lateral drains roughly "
             "200 acres and recovers 3-15% of oil in place."),
    "Vertical gas (conventional)": dict(
        product="gas", area=160.0, rf=70.0, rf_typical=(0.50, 0.85), rf_plausible=(0.30, 0.95),
        label="a conventional gas well", b=0.5, d_eff=25.0, q_limit=25.0, ratio=0.005,
        fvf=0.005, capex=2_000_000.0, loe_fixed=3000.0, loe_var=2.0, loe_var_gas=0.30,
        oil_diff=-5.00, gas_realized=90.0,
        note="A vertical well in a conventional gas sand drains roughly 160 acres. Gas "
             "expands as pressure drops, so 50-85% of gas in place is typically recovered."),
    "Horizontal gas (shale)": dict(
        product="gas", area=100.0, rf=25.0, rf_typical=(0.10, 0.35), rf_plausible=(0.05, 0.50),
        label="a horizontal shale gas well", b=1.2, d_eff=65.0, q_limit=50.0, ratio=0.002,
        fvf=0.0045, capex=5_500_000.0, loe_fixed=6000.0, loe_var=2.0, loe_var_gas=0.50,
        oil_diff=-5.00, gas_realized=90.0,
        note="A fractured horizontal shale gas well drains roughly 80-120 acres and "
             "recovers 10-35% of gas in place."),
}

# Units and wording for each main product
UNITS = {
    "oil": dict(rate="bbl/day", vol="bbl", in_place="Oil in place", abbr="OOIP", price="WTI",
                price_unit="$/bbl", other="Gas-oil ratio", other_unit="scf/bbl", other_scale=1000,
                other_step=100.0, fvf_label="Oil formation volume factor, Bo (rb/stb)", fvf_step=0.05,
                fvf_help="Reservoir barrels per stock tank barrel. Oil shrinks as gas leaves it at the surface.",
                formula="OOIP = 7758 × A × h × φ × (1 − Sw) / Bo", sens=np.arange(30, 125, 5.0)),
    "gas": dict(rate="Mcf/day", vol="Mcf", in_place="Gas in place", abbr="OGIP", price="Henry Hub",
                price_unit="$/MCF", other="Condensate yield", other_unit="bbl/MMcf", other_scale=1000,
                other_step=1.0, fvf_label="Gas formation volume factor, Bg (rcf/scf)", fvf_step=0.0005,
                fvf_help="Reservoir cubic feet per standard cubic foot. Gas expands roughly 100-250× "
                         "on its way to the surface, so Bg is small.",
                formula="OGIP = 43,560 × A × h × φ × (1 − Sw) / Bg", sens=np.arange(1.0, 8.5, 0.5)),
}

# Appraisal modes (sidebar step 3) and where a lease comes from (step 4)
TYPICAL = "Prospect: typical assumptions"
ANALOG = "Prospect: decline from an analog well"
OWN = "Producing well: its own production"
EXAMPLE, BY_API, BY_LEASE = "Example well", "Find by API number", "Enter RRC lease / gas well ID"
MODE_HELP = {
    TYPICAL: "No production data. EUR = volume in place × recovery factor, with a typical "
             "decline shape for the well type.",
    ANALOG: "A similar nearby well's production sets the decline shape and the ratio of the "
            "other stream. Your log still sets the size: volume in place × recovery factor.",
    OWN: "This well's own production history sets the forecast. The recovery factor "
         "(EUR ÷ volume in place) becomes a cross-check between the log and the production.",
}
PRICE_FLAT, PRICE_ESC, PRICE_CUSTOM = "Flat", "Escalating", "Custom deck"
MATRIX = {"Sandstone": 2.65, "Limestone": 2.71, "Dolomite": 2.87}  # grain density, g/cc

# Log sources (sidebar step 1). The real example's zone and lithology come from the
# KGS well page (formation tops from the geologist's report).
SRC_SAMPLE, SRC_REAL, SRC_UPLOAD = ("Synthetic test well", "Real example: Nuss 1, Kansas", "Upload a LAS file")
REAL_LAS = ROOT / "data" / "logs" / "nuss_1.las"
LOG_PRESETS = {
    SRC_REAL: dict(top=3106.0, base=3342.0, lithology="Limestone", gr_clean=15.0, gr_shale=100.0,
                   note="Nuss 1 (API 15-009-25378), Barton County, Kansas: discovery well of the Galatia "
                        "East field, producing from the Lansing-Kansas City limestones (3,106-3,342 ft). "
                        "Log courtesy of the Kansas Corporation Commission via the Kansas Geological Survey."),
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


def volume(x, product):
    """Volume in the main product's units: bbl for oil, Mcf for gas."""
    if product == "gas":
        return f"{x/1e6:,.2f} Bcf" if x >= 1e6 else f"{x/1e3:,.0f} MMcf"
    return f"{x/1e6:,.2f} MMbbl" if x >= 1e6 else f"{x/1e3:,.0f} Mbbl"


def lease_id(lease_no, kind):
    """RRC lease number as printed: gas well IDs are 6 digits with leading zeros."""
    return str(lease_no).zfill(6) if kind == "Gas" else str(lease_no)


@st.cache_data
def load_data():
    examples = {"oil": load_production(ROOT / "data" / "spraberry_leases.csv"),
                "gas": load_production(ROOT / "data" / "barnett_gas_wells.csv")}
    prices = {"oil": load_wti(ROOT / "data" / "wti_monthly.csv"),
              "gas": load_henry_hub(ROOT / "data" / "henry_hub_monthly.csv")}
    return examples, prices


@st.cache_data(ttl=86400, show_spinner="Fetching production from the Railroad Commission...")
def fetch_cached(lease_no, district, kind):
    return clean_production(fetch_lease(lease_no, district, kind=kind))


@st.cache_data(ttl=86400, show_spinner="Looking up the API number at the Railroad Commission...")
def lookup_cached(api8):
    return lookup_api(api8)


@st.cache_data(ttl=86400, show_spinner="Looking up the API number at the Kansas Geological Survey...")
def kgs_lookup_cached(api_digits):
    return kgs.lookup_api(api_digits, index_path=ROOT / "data" / "raw" / "all_kansas_leases.zip")


@st.cache_data(ttl=86400, show_spinner="Fetching production from the Kansas Geological Survey...")
def kgs_fetch_cached(lease_kid):
    sales = kgs.fetch_lease(lease_kid)
    return kgs.spread_sales(sales), kgs.wells_on_lease(sales)


def api_target(text):
    """('TX', 8-digit API) or ('KS', full digits) for a Texas or Kansas API number, else None."""
    parts = kgs.split_api(text)
    if parts and parts[0] == "15":
        return "KS", "".join(parts)
    api8 = normalize_api(text)
    return ("TX", api8) if api8 else None


@st.cache_data(ttl=86400, show_spinner=False)
def well_count(lease_no, district, kind):
    try:
        return len(lease_wells(lease_no, district, kind)) or 1
    except Exception:  # count is a nicety; don't fail the page over it
        return 1


@st.cache_data
def read_las(text):
    las = lasio.read(text)
    df = las.df().reset_index()
    df = df.rename(columns={df.columns[0]: "DEPT"})
    header = {item.mnemonic: str(item.value) for item in las.well}
    params = {item.mnemonic: item.value for item in las.params}
    units = {c.mnemonic: c.unit for c in las.curves}
    return df, header, params, units


def warn_if_poor_fit(fit, whose):
    """Flag a decline fit that one Arps curve doesn't describe well."""
    problems = []
    if fit.b >= 1.99:
        problems.append("the b-factor hit its 2.0 limit")
    if fit.rmse_log > 0.35:
        problems.append("the months scatter widely around the curve")
    if problems:
        st.warning(f"{whose} decline isn't well described by one curve: {' and '.join(problems)}. "
                   "Long histories often include workovers or refracs. Check the chart, and try a later "
                   "fit start or set the shape by hand.", icon="⚠️")


def pick_curve(label, key, columns):
    default = next((c for c in ALIASES[key] if c in columns), columns[0])
    return st.selectbox(label, columns, index=columns.index(default), key=f"curve_{key}")


examples, hist_prices = load_data()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Well log")
    source = st.radio("LAS file", [SRC_SAMPLE, SRC_REAL, SRC_UPLOAD], label_visibility="collapsed")
    if source == SRC_UPLOAD:
        up = st.file_uploader("LAS file", type=["las", "LAS"])
        las_text = up.getvalue().decode("utf-8", errors="ignore") if up else None
    elif source == SRC_REAL:
        las_text = REAL_LAS.read_text()
    else:
        las_text = SAMPLE_LAS.read_text()
    if las_text is None:
        st.info("Upload a LAS file, or pick one of the examples.")
        st.stop()
    preset = LOG_PRESETS.get(source, {})
    if preset:
        st.caption(preset["note"])
    log, header, las_params, units = read_las(las_text)
    api = (header.get("API") or "").strip() or (header.get("UWI") or "").strip()
    st.caption(f"Log: {header.get('WELL', 'unnamed well')}" + (f" · API {api}" if api else ""))

    st.header("2 · Well type")
    well_type = st.radio("Well type", list(WELL_TYPES), label_visibility="collapsed")
    wt = WELL_TYPES[well_type]
    product = wt["product"]
    P, U = PRODUCTS[product], UNITS[product]
    kind = "Oil" if product == "oil" else "Gas"
    st.caption(wt["note"] + " Sets starting assumptions; all are editable.")

    st.header("3 · Appraisal mode")
    mode = st.radio("Appraisal mode", [TYPICAL, ANALOG, OWN], label_visibility="collapsed")
    st.caption(MODE_HELP[mode])

    lease, lease_label, n_wells = None, "", 1
    prod_source = None
    if mode != TYPICAL:
        st.header("4 · " + ("Analog well" if mode == ANALOG else "This well's production"))
        sources = [EXAMPLE, BY_API, BY_LEASE]
        # A producing well's own log usually carries its API number: look that up first
        default_src = BY_API if mode == OWN and api_target(api) else EXAMPLE
        prod_source = st.radio("Production source", sources, index=sources.index(default_src),
                               label_visibility="collapsed", key=f"src_{mode}")

    if prod_source == EXAMPLE:
        ex = examples[product]
        wells = ex.groupby(["lease_no", "lease_name"]).size().reset_index().sort_values("lease_name")
        labels = [f"{r.lease_name} ({lease_id(r.lease_no, kind)})" for r in wells.itertuples()]
        choice = st.selectbox("Spraberry oil lease" if product == "oil" else "Barnett Shale gas well", labels)
        lease = ex[ex["lease_no"] == int(wells.iloc[labels.index(choice)]["lease_no"])]

    elif prod_source == BY_API:
        with st.form("api_form"):
            api_in = st.text_input("API number", value=api if mode == OWN else "",
                                   placeholder="e.g. 42-227-40326",
                                   help="Texas (42-...) or Kansas (15-...). Read from the log header "
                                        "(API or UWI) when present.")
            submitted = st.form_submit_button("Look up production", width="stretch")
        if submitted:
            target = api_target(api_in)
            if target:
                st.session_state["api_lookup"] = target
            else:
                st.error("Enter a Texas (42-CCC-WWWWW) or Kansas (15-CCC-WWWWW) API number.")
        target = st.session_state.get("api_lookup")
        if target and target[0] == "KS":
            try:
                found = kgs_lookup_cached(target[1])
                if not found:
                    st.error("The KGS has no oil lease for that API number (the well may be a dry hole "
                             "or a gas well). Run `python fetch_data.py` once to download the KGS "
                             "lease index if you haven't.")
                else:
                    w = found[0]
                    well_txt = f" #{w['well']}" if w["well"] else ""
                    st.caption(f"API {w['api']} → {w['lease_name']}{well_txt}, KGS oil lease "
                               f"{w['lease_kid']}, {w['county']} County, Kansas")
                    lease, n_wells = kgs_fetch_cached(w["lease_kid"])
                    st.caption("Kansas reports oil sales (truckloads), not production: each year's sales "
                               "are spread evenly over its producing months. No casinghead gas is reported.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # network trouble, site changes
                st.error(f"Couldn't reach the Kansas Geological Survey: {e}")
        elif target:
            try:
                found = lookup_cached(target[1])
                if not found:
                    st.error("The RRC has no wellbore with that API number.")
                else:
                    w = found[0]
                    st.caption(f"API {w['api'][:3]}-{w['api'][3:]} → {w['lease_name']} #{w['well_no']}, "
                               f"{'oil lease' if w['kind'] == 'Oil' else 'gas well ID'} "
                               f"{w['lease_no']} (District {w['district']}), {w['county'].title()} County")
                    lease = fetch_cached(w["lease_no"], w["district"], w["kind"])
                    if w["kind"] == "Oil":
                        n_wells = well_count(w["lease_no"], w["district"], "Oil")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # network trouble, site changes
                st.error(f"Couldn't reach the Railroad Commission: {e}")

    elif prod_source == BY_LEASE:
        with st.form("fetch"):
            c1, c2 = st.columns([3, 2])
            lease_in = c1.text_input("Lease number or gas well ID", placeholder="e.g. 50678",
                                     help="Oil leases have 5-digit numbers; gas wells have 6-digit IDs.")
            district = c2.selectbox("District", list(DISTRICT_CODES), index=list(DISTRICT_CODES).index("08"))
            submitted = st.form_submit_button("Fetch from RRC", width="stretch")
        if submitted:
            if lease_in.strip().isdigit():
                st.session_state["fetched_lease"] = (lease_in.strip(), district)
            else:
                st.error("Lease number must be digits only.")
        if "fetched_lease" in st.session_state:
            lease_no, district = st.session_state["fetched_lease"]
            try:
                lease = fetch_cached(lease_no, district, lease_kind(lease_no))
                if lease_kind(lease_no) == "Oil":
                    n_wells = well_count(lease_no, district, "Oil")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Couldn't reach the Railroad Commission: {e}")
        st.caption("The number is on the lease sign and drilling permits. Or look the well up by "
                   "its API number instead.")

    if lease is not None:
        lease_kind_ = lease["kind"].iloc[0]
        lease_label = f"{lease['lease_name'].iloc[0]} ({lease_id(lease['lease_no'].iloc[0], lease_kind_)})"
        own_rate = PRODUCTS["oil" if lease_kind_ == "Oil" else "gas"]["rate"]
        producing_months = int((lease[own_rate] > 0).sum())
        op = lease["operator"].dropna()
        st.caption(f"{lease_label} · {lease['field'].iloc[0]} · "
                   f"{op.iloc[-1] if len(op) else 'operator n/a'} · {producing_months} producing months")
        if lease_kind_ != kind:
            st.error(f"This is {'an oil lease' if lease_kind_ == 'Oil' else 'a gas well'}, but the well type "
                     f"is {product}. Pick a matching well type in step 2.", icon="⛔")
            lease = None
        elif producing_months < 6:
            st.warning("Fewer than 6 producing months: too little history to fit a decline. "
                       "Using typical assumptions instead.", icon="⚠️")
            lease = None
        elif mode == OWN and prod_source != BY_API:
            st.caption("Make sure this production is from the well the log was run in; otherwise "
                       "the cross-check is meaningless. Looking it up by API number avoids mix-ups.")

    if lease is not None and mode == OWN and n_wells > 1:
        st.warning(f"This oil lease has {n_wells} wells. Oil is reported for the whole lease, "
                   "so its production may come from several wells.", icon="⚠️")
        share = st.number_input("Wells sharing the production", min_value=1, value=n_wells, step=1,
                                help="Lease volumes are divided equally between this many wells. "
                                     "Set to 1 if the other wellbores never produced.")
        if share > 1:
            lease = lease.copy()
            for c in ("oil_bbl", "gas_mcf", "bopd", "gas_mcfd"):
                lease[c] = lease[c] / share
            lease_label += f", 1/{share} share"

has_prod = mode == OWN and lease is not None      # forecast from this well's own history
use_analog = mode == ANALOG and lease is not None  # decline shape borrowed from another well

summary_tab, log_tab, vol_tab, fc_tab, econ_tab = st.tabs(
    ["Summary", "1 · Well log", "2 · Volumetrics", "3 · Production forecast", "4 · Economics"])

# ---------------------------------------------------------------- 1. well log
with log_tab:
    log_key = f"{header.get('WELL', '')}_{len(log)}"  # a new log resets the zone
    d_min, d_max = float(log["DEPT"].min()), float(log["DEPT"].max())
    with st.expander("Interpretation parameters", expanded=source == SRC_UPLOAD):
        cols = [c for c in log.columns if c != "DEPT"]
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            st.markdown("**Zone to evaluate**")
            z_top = st.number_input("Top (ft)", value=preset.get("top", d_min), min_value=d_min, max_value=d_max,
                                    step=10.0,
                                    key=f"ztop_{log_key}",
                                    help="Limit the evaluation to the producing formation. A whole-well log "
                                         "includes salt, anhydrite and other zones that can falsely pass "
                                         "the pay cutoffs.")
            z_base = st.number_input("Base (ft)", value=preset.get("base", d_max), min_value=d_min,
                                     max_value=d_max, step=10.0, key=f"zbase_{log_key}")
        with c1:
            st.markdown("**Curves**")
            gr_col = pick_curve("Gamma ray", "GR", cols)
            rhob_col = pick_curve("Bulk density", "RHOB", cols)
            rt_col = pick_curve("Deep resistivity", "RT", cols)
        with c2:
            st.markdown("**Rock and fluid**")
            lithology = st.selectbox("Lithology", list(MATRIX), key=f"lith_{log_key}",
                                     index=list(MATRIX).index(preset.get("lithology", "Sandstone")),
                                     help="Sets the matrix (grain) density.")
            rho_matrix = st.number_input("Matrix density (g/cc)", value=MATRIX[lithology],
                                         key=f"rhoma_{log_key}_{lithology}",
                                         help="2.65 sandstone, 2.71 limestone, 2.87 dolomite")
            gr_clean = st.number_input("GR clean (API)", value=preset.get("gr_clean", 25.0), key=f"grc_{log_key}")
            gr_shale = st.number_input("GR shale (API)", value=preset.get("gr_shale", 110.0), key=f"grs_{log_key}")
            rho_fluid = st.number_input("Fluid density (g/cc)", value=1.0)
            rw_default = float(las_params.get("RW", 0.05) or 0.05)
            rw = st.number_input("Rw, formation water resistivity (ohm-m)", value=rw_default,
                                 format="%.3f", help="Read from the LAS ~Parameter section when present")
        with c3:
            st.markdown("**Pay cutoffs**")
            vsh_max = st.number_input("Shale volume ≤", value=0.40, step=0.05)
            phi_min = st.number_input("Porosity ≥", value=0.08, step=0.01)
            sw_max = st.number_input("Water saturation ≤", value=0.50, step=0.05)

    if z_base <= z_top:
        st.error("The zone base must be deeper than its top.")
        st.stop()
    log = log[(log["DEPT"] >= z_top) & (log["DEPT"] <= z_base)].copy()
    if len(log) < 2:
        st.error("The zone holds fewer than two log samples; widen it.")
        st.stop()
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
        shown = ", ".join(f"{d:g} ft" for d in missing["DEPT"].head(20))
        more = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
        st.warning(f"Missing data at {len(missing)} depth(s), not counted as pay: {shown}{more}", icon="⚠️")

    tracks = [(gr_col, "GR (API)", BLUE, False), (rt_col, "Rt (ohm-m)", BLUE, True),
              ("PHI", "Porosity", BLUE, False), ("SW", "Water sat.", BLUE, False)]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[t[1] for t in tracks])
    step = float(np.median(np.diff(log["DEPT"])))
    for i, (col, name, color, logx) in enumerate(tracks, start=1):
        fig.add_trace(go.Scatter(x=log[col], y=log["DEPT"], mode="lines", name=name,
                                 line=dict(color=color, width=2 if len(log) < 2000 else 1), showlegend=False,
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
    style(fig, height=560 if len(log) < 2000 else 800).update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded green: pay, where all three cutoffs pass.")
    with st.expander("Pay zone table"):
        st.dataframe(log.loc[log["PAY"], ["DEPT", "VSH", "PHI", "SW"]].round(3), hide_index=True)

# ---------------------------------------------------------------- 2. volumetrics
with vol_tab:
    st.markdown(f"{U['in_place']}: **{U['formula']}**")
    c1, c2 = st.columns(2)
    area = c1.number_input("Drainage area, A (acres)", value=wt["area"], step=10.0, key=f"area_{well_type}",
                           help="The area one well drains. Not in the log; set by well type and spacing.")
    fvf = c2.number_input(U["fvf_label"], value=wt["fvf"], step=U["fvf_step"], key=f"fvf_{well_type}",
                          format="%.4f" if product == "gas" else "%.2f", help=U["fvf_help"])
    if not pay["net_pay_ft"]:
        in_place = 0.0
    elif product == "oil":
        in_place = ooip(area, pay["net_pay_ft"], pay["avg_phi"], pay["avg_sw"], fvf)        # STB
    else:
        in_place = ogip(area, pay["net_pay_ft"], pay["avg_phi"], pay["avg_sw"], fvf) / 1000  # Mcf
    m = st.columns(4)
    m[0].metric(U["abbr"], volume(in_place, product))
    m[1].metric("Net pay, h", f"{pay['net_pay_ft']:.1f} ft")
    m[2].metric("Porosity, φ", f"{pay['avg_phi']:.1%}" if pay["net_pay_ft"] else "n/a")
    m[3].metric("Water saturation, Sw", f"{pay['avg_sw']:.1%}" if pay["net_pay_ft"] else "n/a")
    st.caption("h, φ and Sw come from the well log tab. Sw is averaged by pore volume.")

# ---------------------------------------------------------------- 3. production forecast
rate_unit = U["rate"]
with fc_tab:
    if has_prod:
        st.markdown(f"**Decline curve fitted to actual production** · {lease_label}")
        producing = lease[lease[P["rate"]] > 0]
        peak_date = producing.loc[producing[P["rate"]].idxmax(), "date"]
        dates = list(producing.loc[producing["date"] >= peak_date, "date"])

        c1, c2, c3, c4 = st.columns(4)
        start = c1.selectbox("Fit from", dates, format_func=lambda d: d.strftime("%b %Y"),
                             help="Defaults to peak month. Move it later to fit only the decline "
                                  "after a workover or refrac.")
        model = c2.radio("Forecast model", MODELS, index=MODELS.index("hyperbolic"))
        d_term = c3.number_input("Terminal decline (%/yr)", value=8.0, step=1.0,
                                 help="Hyperbolic decline switches to exponential at this rate, "
                                      "so b > 1 can't forecast infinite reserves.") / 100
        q_limit = c4.number_input(f"Forecast cutoff ({rate_unit})", value=wt["q_limit"], step=1.0,
                                  key=f"ql_own_{well_type}",
                                  help="Forecast stops below this rate. The economics tab applies "
                                       "its own economic limit on top.")

        hist = prepare_history(lease, start, rate=P["rate"])
        if len(hist) < 4:
            st.error("Fewer than 4 producing months after the chosen start; pick an earlier month.")
            st.stop()
        fits = {mdl: fit_arps(hist, mdl) for mdl in MODELS}
        forecasts = {mdl: forecast(f, hist, q_limit=q_limit, d_term_annual=d_term) for mdl, f in fits.items()}
        fit, fc = fits[model], forecasts[model]
        well_eur = eur(lease, fc, P["volume"])
        cum = float(lease[P["volume"]].sum())

        m = st.columns(4)
        m[0].metric("EUR", volume(well_eur, product))
        m[1].metric("Produced to date", volume(cum, product))
        m[2].metric("Remaining (forecast)", volume(well_eur - cum, product))
        m[3].metric("Initial decline / b", f"{fit.di_annual_eff:.0%}/yr · b={fit.b:.2f}")
        warn_if_poor_fit(fit, "This well's")

        fig = go.Figure()
        excluded = lease[(lease["date"] < hist["date"].iloc[0]) | ~(lease[P["rate"]] > 0)]
        fig.add_trace(go.Scatter(x=excluded["date"], y=excluded[P["rate"]].clip(lower=0.5), mode="markers",
                                 name="Not fitted", marker=dict(color=MUTED, size=8, symbol="circle-open")))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["q"], mode="markers", name="Actual (fitted)",
                                 marker=dict(color=BLUE, size=8 if len(hist) < 150 else 5,
                                             line=dict(color="white", width=1))))
        for mdl in MODELS:
            f = fits[mdl]
            t_all = np.arange(0, int(forecasts[mdl]["t"].max() if len(forecasts[mdl]) else hist["t"].max()) + 1)
            d_all = [f.t0 + pd.DateOffset(months=int(t)) for t in t_all]
            q_all = f.rate(t_all, nominal_from_effective_annual(d_term))
            fig.add_trace(go.Scatter(x=d_all, y=q_all, mode="lines", name=f"{mdl.capitalize()}",
                                     line=dict(color=MODEL_COLOR[mdl], width=3 if mdl == model else 2,
                                               dash=None if mdl == model else "dot")))
        fig.add_vline(x=lease["date"].max(), line=dict(color=MUTED, width=1, dash="dash"))
        fig.update_yaxes(type="log", title_text=f"{product.capitalize()} rate ({rate_unit}, log scale)")
        style(fig, height=480)
        st.plotly_chart(fig, width="stretch")
        st.caption("Dashed vertical line: last reported month. Solid curve: selected forecast model.")

        table = pd.DataFrame([{
            "Model": mdl.capitalize(), f"qi ({rate_unit})": round(f.qi), "Di (%/yr)": f"{f.di_annual_eff:.0%}",
            "b": round(f.b, 2), "Fit error (log RMSE)": round(f.rmse_log, 2),
            f"EUR ({'Mbbl' if product == 'oil' else 'MMcf'})": round(eur(lease, forecasts[mdl], P["volume"]) / 1e3)}
            for mdl, f in fits.items()])
        st.dataframe(table, hide_index=True)
        st.caption("Fit uses a robust loss that down-weights months with partial downtime. "
                   "Zero and unreported months are left out.")
    else:
        model = "type curve"
        b_default, deff_default, key_sfx = wt["b"], wt["d_eff"], well_type
        if use_analog:
            # Borrow the analog's decline shape: fit it, then use its b and first-year decline
            analog_hist = prepare_history(lease, rate=P["rate"])
            analog_fit = fit_arps(analog_hist, "hyperbolic")
            analog_fc = forecast(analog_fit, analog_hist, q_limit=wt["q_limit"])
            b_default = float(np.clip(round(analog_fit.b, 2), 0.0, 2.0))
            deff_default = float(np.clip(round(analog_fit.di_annual_eff * 100), 1, 99))
            key_sfx = f"{well_type}_{lease['lease_no'].iloc[0]}"  # new analog -> fresh defaults
            st.markdown(f"**Forecast from the log, shaped like analog {lease_label}**: "
                        f"your {U['in_place'].lower()} × recovery factor sets how much; the analog's "
                        "actual decline sets how fast.")
            m = st.columns(4)
            m[0].metric("Analog EUR", volume(eur(lease, analog_fc, P["volume"]), product))
            m[1].metric("Analog peak rate", f"{analog_hist['q'].iloc[0]:,.0f} {rate_unit}")
            m[2].metric("Analog b-factor", f"{analog_fit.b:.2f}")
            m[3].metric("Analog first-year decline", f"{analog_fit.di_annual_eff:.0%}")
            warn_if_poor_fit(analog_fit, "The analog's")
        else:
            st.markdown(f"**Forecast from the log** (no production data): EUR = {U['in_place'].lower()} × "
                        "recovery factor, spread over time with a typical decline shape for the well type.")
        c1, c2, c3 = st.columns(3)
        rf_assumed = c1.number_input("Recovery factor (%)", value=wt["rf"], step=1.0, key=f"rf_{well_type}",
                                     help="Share of the volume in place the well will recover. The biggest "
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
        q_limit = c3.number_input(f"Forecast cutoff ({rate_unit})", value=wt["q_limit"], step=1.0,
                                  key=f"ql_{well_type}")

        well_eur = in_place * rf_assumed
        try:
            fit, fc = type_curve(well_eur, pd.Timestamp(first_prod).replace(day=1), b, d_eff,
                                 d_term_annual=d_term, q_limit=q_limit)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        m = st.columns(4)
        m[0].metric("EUR", volume(well_eur, product))
        m[1].metric("Initial rate", f"{fit.qi:,.0f} {rate_unit}" if len(fc) else "n/a")
        m[2].metric("First-year volume", volume(float(fc["volume"].head(12).sum()), product))
        m[3].metric("Producing life", f"{len(fc)/12:.1f} years")
        if fc.empty:
            st.warning(f"{U['in_place']} × recovery factor is too small to produce above the forecast "
                       "cutoff rate at all.", icon="⚠️")
        else:
            # Months on production, so a historical analog lines up with a future well
            fig = go.Figure()
            hover = f"Month %{{x}}: %{{y:,.0f}} {rate_unit}<extra></extra>"
            if use_analog:
                fig.add_trace(go.Scatter(x=analog_hist["t"], y=analog_hist["q"], mode="markers",
                                         name=f"Analog actual ({lease_label})",
                                         marker=dict(color=BLUE, size=8 if len(analog_hist) < 150 else 5,
                                                     line=dict(color="white", width=1)),
                                         hovertemplate=hover))
                t_a = np.arange(0, int(max(analog_fc["t"].max() if len(analog_fc) else 0, analog_hist["t"].max())) + 1)
                fig.add_trace(go.Scatter(x=t_a, y=analog_fit.rate(t_a, nominal_from_effective_annual(d_term)),
                                         mode="lines", name="Analog fit", line=dict(color=BLUE, width=2, dash="dot"),
                                         hovertemplate=hover))
            fig.add_trace(go.Scatter(x=fc["t"], y=fc["q"], mode="lines", name="Your well (forecast)",
                                     line=dict(color=ORANGE, width=3), hovertemplate=hover))
            fig.update_xaxes(title_text="Months on production")
            fig.update_yaxes(type="log", title_text=f"{product.capitalize()} rate ({rate_unit}, log scale)")
            style(fig, height=440).update_layout(hovermode="closest", showlegend=use_analog)
            st.plotly_chart(fig, width="stretch")
            if use_analog:
                st.caption("Same shape, different size: your forecast follows the analog's decline, "
                           f"scaled so it recovers your log's {U['in_place'].lower()} × recovery factor.")
        st.info("This forecast is only as good as its assumptions. If this well already produces, "
                "switch to 'Producing well' mode to use its actual decline.", icon="ℹ️")

# ---------------------------------------------------------------- 4. economics
with econ_tab:
    wti_hist, hh_hist = hist_prices["oil"], hist_prices["gas"]
    recent = {"oil": float(wti_hist["wti"].tail(12).mean()), "gas": float(hh_hist["hh"].tail(12).mean())}
    first_year = int(fc["date"].iloc[0].year) if len(fc) else pd.Timestamp.today().year

    with st.expander("Price outlook", expanded=False):
        deck_kind = st.radio("Forecast prices", [PRICE_FLAT, PRICE_ESC, PRICE_CUSTOM], horizontal=True,
                             help="Months already produced always use actual historical prices (FRED).")
        c1, c2, c3 = st.columns(3)
        if deck_kind == PRICE_CUSTOM:
            n_years = 10
            default_deck = pd.DataFrame({"Year": np.arange(first_year, first_year + n_years),
                                         "WTI ($/bbl)": round(recent["oil"], 2),
                                         "Henry Hub ($/MCF)": round(recent["gas"], 2)})
            deck = st.data_editor(default_deck, hide_index=True, num_rows="dynamic", key=f"deck_{first_year}")
            deck = deck.dropna().sort_values("Year")
            if deck.empty:
                st.error("The price deck needs at least one year.")
                st.stop()
            years = deck["Year"].astype(int).to_numpy()
            oil_deck, gas_deck = deck["WTI ($/bbl)"].to_numpy(float), deck["Henry Hub ($/MCF)"].to_numpy(float)
        else:
            wti0 = c1.number_input("WTI ($/bbl)", value=round(recent["oil"], 2), step=1.0,
                                   help="Defaults to the last 12 months' average WTI (FRED).")
            hh0 = c2.number_input("Henry Hub ($/MCF)", value=round(recent["gas"], 2), step=0.25,
                                  help="Defaults to the last 12 months' average Henry Hub (FRED).")
            esc = c3.number_input("Escalation (%/yr)", value=2.5, step=0.5,
                                  help="Prices grow this much per year for 10 years, then stay flat.") / 100 \
                if deck_kind == PRICE_ESC else 0.0
            n_years = 10 if deck_kind == PRICE_ESC else 1
            years, oil_deck = escalated_deck(first_year, n_years, wti0, esc)
            _, gas_deck = escalated_deck(first_year, n_years, hh0, esc)
        if deck_kind != PRICE_FLAT:
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                                subplot_titles=["WTI ($/bbl)", "Henry Hub ($/MCF)"])
            for i, (deck_vals, fmt) in enumerate([(oil_deck, "$.0f"), (gas_deck, "$.2f")], start=1):
                fig.add_trace(go.Scatter(x=years, y=deck_vals, mode="lines+markers", showlegend=False,
                                         line=dict(color=BLUE, width=2), marker=dict(size=7),
                                         hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>"), row=1, col=i)
                fig.update_yaxes(tickformat=fmt, rangemode="tozero", row=1, col=i)
                fig.update_xaxes(dtick=2, row=1, col=i)
            style(fig, height=260).update_layout(hovermode="closest")
            st.plotly_chart(fig, width="stretch")
            st.caption("After the last deck year, prices stay flat.")

    with st.expander("Cost and fiscal assumptions", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        liquid = "oil" if product == "oil" else "condensate"
        a = Assumptions(
            oil_price=float(oil_deck[0]), gas_price=float(gas_deck[0]),
            oil_diff=c1.number_input(f"{liquid.capitalize()} differential to WTI ($/bbl)", value=wt["oil_diff"],
                                     step=0.25, key=f"odiff_{well_type}"),
            gas_diff_pct=c1.number_input("Gas realized (% of Henry Hub)", value=wt["gas_realized"], step=5.0,
                                         key=f"gdiff_{well_type}",
                                         help="Casinghead gas from oil wells sells at a bigger discount "
                                              "than pipeline-quality gas-well gas.") / 100,
            working_interest=c2.number_input("Working interest (%)", value=100.0, step=5.0) / 100,
            royalty=c2.number_input("Royalty (%)", value=25.0, step=1.0) / 100,
            severance_oil=c2.number_input("Oil severance tax (%)", value=4.6, step=0.1) / 100,
            severance_gas=c2.number_input("Gas severance tax (%)", value=7.5, step=0.1) / 100,
            ad_valorem=c3.number_input("Ad valorem tax (%)", value=2.0, step=0.5) / 100,
            loe_fixed=c3.number_input("Fixed LOE ($/month)", value=wt["loe_fixed"], step=500.0,
                                      key=f"loef_{well_type}"),
            loe_var=c3.number_input(f"Variable LOE ($/bbl {liquid})", value=wt["loe_var"], step=0.5,
                                    key=f"loev_{well_type}"),
            loe_var_gas=c3.number_input("Variable LOE ($/Mcf gas)", value=wt["loe_var_gas"], step=0.05,
                                        key=f"loeg_{well_type}", help="Compression, gathering, treating."),
            capex=c4.number_input("Drill & complete capex ($)", value=wt["capex"], step=250_000.0,
                                  key=f"capex_{well_type}"),
            discount_rate=c4.number_input("Discount rate (%/yr)", value=10.0, step=1.0) / 100,
        )
        if not has_prod:
            # A new well's whole life is being forecast, so use the analog's lifetime ratio
            # (an oil well's late-life GOR rises as reservoir pressure drops and would overstate gas)
            ratio_default = secondary_ratio(lease, product, months=None) if use_analog else wt["ratio"]
            ratio = c4.number_input(f"{U['other']} ({U['other_unit']})",
                                    value=float(round(ratio_default * U["other_scale"], 1)),
                                    step=U["other_step"], key=f"ratio_{key_sfx}",
                                    help=("Gas produced per barrel of oil" if product == "oil"
                                          else "Condensate produced per million cubic feet of gas")
                                    + (" (analog's lifetime average)." if use_analog else ".")) / U["other_scale"]

    rate_pct = f"{a.discount_rate:.0%}"
    if has_prod:
        vol = full_volumes(lease, fc, product)
    else:
        vol = forecast_volumes(fc, ratio, product)
        if vol.empty:  # nothing to produce: the well still costs its capex
            vol = pd.DataFrame({"date": [pd.Timestamp(first_prod).replace(day=1)], "oil_bbl": [0.0],
                                "gas_mcf": [0.0], "actual": [False]})
    # Monthly prices: actual history for produced months, the deck for forecast months
    oil_fc_prices = price_deck(vol["date"], years, oil_deck)
    gas_fc_prices = price_deck(vol["date"], years, gas_deck)
    oil_prices = priced_history(vol, wti_hist, oil_fc_prices)
    gas_prices = priced_history(vol, hh_hist, gas_fc_prices)
    main_prices, other_kw = ((oil_prices, "gas_prices") if product == "oil" else (gas_prices, "oil_prices"))
    other_prices = gas_prices if product == "oil" else oil_prices
    main_hist = wti_hist if product == "oil" else hh_hist

    def priced_flat(p, mask=slice(None)):
        """Prices with the main product flat at p for forecast months (history kept)."""
        main = priced_history(vol, main_hist, p)
        kw = {"oil_prices": main, "gas_prices": gas_prices} if product == "oil" else \
             {"oil_prices": oil_prices, "gas_prices": main}
        return {k: v[mask] for k, v in kw.items()}

    full_cf = cash_flow(vol, a, oil_prices=oil_prices, gas_prices=gas_prices, include_capex=True)
    full = summarize(full_cf)
    breakeven = breakeven_price(vol, a, product, **{other_kw: other_prices})
    be_label = f"Breakeven {U['price']} ({U['price_unit']})"
    be_text = (f"${breakeven:,.2f}" if not np.isnan(breakeven)
               else f"> ${BREAKEVEN_MAX[product]:,.0f}")

    if has_prod:
        st.subheader("Full-cycle: was drilling this well worth it?")
        st.caption("From first production, including capex. Months already produced use actual "
                   "historical prices; forecast months use the price outlook.")
    else:
        st.subheader("Drilling decision: is this well worth drilling?")
        st.caption("The log-based forecast at the price outlook, including capex.")
    m = st.columns(4)
    m[0].metric(f"NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    m[2].metric("Payout", f"{full.payout_months:.0f} mo" if not np.isnan(full.payout_months) else "never")
    m[3].metric(be_label, be_text, help=f"Flat {U['price']} price over the well's life that gives NPV = 0.")

    pdp = None
    if has_prod:
        future = ~vol["actual"].to_numpy()
        pdp_vol = vol[future]
        pdp = summarize(cash_flow(pdp_vol, a, oil_prices=oil_prices[future], gas_prices=gas_prices[future]))
        st.subheader("Remaining value: what is the producing well worth today?")
        st.caption("Forecast only (proved developed producing, PDP), at the price outlook, no capex.")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"PDP value @ {rate_pct}", money(pdp.npv))
        remaining = float(cash_flow(pdp_vol, a, oil_prices=oil_prices[future],
                                    gas_prices=gas_prices[future])[P["volume"]].sum())
        c2.metric(f"Remaining economic {product}", volume(remaining, product))
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

    sens = U["sens"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens, y=[cash_flow(vol, a, include_capex=True, **priced_flat(p))["pv"].sum() / 1e6 for p in sens],
        mode="lines+markers", name="Full-cycle NPV", line=dict(color=BLUE, width=2), marker=dict(size=8)))
    if has_prod:
        fig.add_trace(go.Scatter(
            x=sens, y=[cash_flow(pdp_vol, a, **priced_flat(p, future))["pv"].sum() / 1e6 for p in sens],
            mode="lines+markers", name="PDP value", line=dict(color=ORANGE, width=2), marker=dict(size=8)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text=f"Forecast {U['price']} ({U['price_unit']}, flat)")
    fig.update_yaxes(title_text=f"Value @ {rate_pct} ($M)")
    style(fig, height=380).update_layout(title=f"{U['price']} price sensitivity", showlegend=has_prod)
    c2.plotly_chart(fig, width="stretch")

    with st.expander("Monthly cash flow table (full-cycle)"):
        st.dataframe(full_cf.assign(date=full_cf["date"].dt.strftime("%Y-%m")).round(2), hide_index=True)

# ---------------------------------------------------------------- summary
with summary_tab:
    st.title("Upstream appraisal")
    deck_text = {PRICE_FLAT: "flat", PRICE_ESC: "escalating", PRICE_CUSTOM: "custom deck"}[deck_kind]
    main_now = oil_deck[0] if product == "oil" else gas_deck[0]
    st.caption(f"Log: {header.get('WELL', 'unnamed')} · {well_type} · "
               + (f"Producing well: {lease_label} · " if has_prod
                  else f"Prospect, decline from analog {lease_label} · " if use_analog
                  else "Prospect, typical assumptions · ") +
               f"{U['price']} ${main_now:,.2f} ({deck_text})")

    st.markdown(f"#### Rock → {product} in place → recovery → dollars")
    m = st.columns(4)
    m[0].metric("Net pay (log)", f"{pay['net_pay_ft']:.1f} ft")
    m[1].metric(U["in_place"], volume(in_place, product))
    m[2].metric("EUR", volume(well_eur, product))
    if has_prod:
        rf, verdict = recovery_factor_check(well_eur, in_place, wt["rf_typical"], wt["rf_plausible"], wt["label"])
        m[3].metric("Recovery factor", f"{rf:.1%}" if not np.isnan(rf) else "n/a")
        (lo, hi), (tlo, thi) = wt["rf_plausible"], wt["rf_typical"]
        if np.isnan(rf) or rf > hi or rf < lo:
            st.error(f"**Cross-check failed.** {verdict}", icon="⛔")
        elif tlo <= rf <= thi:
            st.success(f"**Cross-check passed.** {verdict}", icon="✅")
        else:
            st.warning(f"**Cross-check: review.** {verdict}", icon="⚠️")
    else:
        m[3].metric("Recovery factor (assumed)", f"{rf_assumed:.0%}")
        shape = f"the decline shape of analog {lease_label}" if use_analog else "a typical decline shape"
        st.info(f"EUR is {U['in_place'].lower()} × an assumed {rf_assumed:.0%} recovery factor, produced "
                f"with {shape}. Once the well has its own production, switch to 'Producing well' mode "
                "to test these assumptions against real performance.", icon="ℹ️")

    m = st.columns(4)
    m[0].metric(f"NPV @ {rate_pct}", money(full.npv))
    m[1].metric("IRR", f"{full.irr:.0%}" if not np.isnan(full.irr) else "n/a")
    if has_prod:
        m[2].metric(f"PDP value @ {rate_pct}", money(pdp.npv))
    else:
        m[2].metric("Payout", f"{full.payout_months:.0f} mo" if not np.isnan(full.payout_months) else "never")
    m[3].metric(be_label, be_text)

    if source == SRC_SAMPLE:
        st.info("The synthetic test well is 12 ft thick with 3.5 ft of pay, so it holds very little. "
                "Pick the real Kansas example or upload a LAS file to appraise a real well.", icon="ℹ️")
