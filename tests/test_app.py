"""Run the dashboard headlessly with the bundled example data (no network) and
check it renders without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MODES = ("Prospect: typical assumptions", "Prospect: decline from an analog well",
         "Producing well: its own production", "Prospect: field type curve")
WELL_TYPES = ("Vertical oil (conventional)", "Horizontal oil (tight)",
              "Vertical gas (conventional)", "Horizontal gas (shale)")


def run(well_type, mode, source=None):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[1].set_value(well_type).run()
    at.sidebar.radio[2].set_value(mode).run()
    if source:
        at.sidebar.radio[3].set_value(source).run()
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


@pytest.mark.parametrize("well_type", WELL_TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_dashboard_renders(well_type, mode):
    source = {MODES[0]: None, MODES[3]: "Example field (bundled)"}.get(mode, "Example well")
    at = run(well_type, mode, source=source)
    assert not at.exception, [e.value for e in at.exception]
    metrics = {m.label for m in at.metric}
    in_place = "Oil in place" if "oil" in well_type else "Gas in place"
    assert {in_place, "EUR", "NPV @ 10%"} <= metrics


def test_real_kansas_log():
    """The real Nuss 1 log, evaluated over its Lansing-Kansas City zone as a limestone."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Real example: Nuss 1, Kansas").run()
    at.sidebar.radio[1].set_value("Vertical oil (conventional)").run()
    assert not at.exception
    assert metric(at, "Net pay") == "23.0 ft"
    assert metric(at, "Avg porosity (pay)") == "17.8%"
    assert next(n.value for n in at.number_input if n.label == "Top (ft)") == 3106.0
    assert next(s.value for s in at.selectbox if s.label == "Lithology") == "Limestone"


def test_field_type_curve_sets_shape():
    at = run("Horizontal oil (tight)", MODES[3], source="Example field (bundled)")
    assert not at.exception
    assert metric(at, "Wells in type curve") == "8"
    b_shown = next(n.value for n in at.number_input if n.label == "b-factor")
    assert b_shown == float(metric(at, "P50 b-factor"))


def test_gas_units():
    at = run("Horizontal gas (shale)", MODES[2], source="Example well")
    assert not at.exception
    assert metric(at, "EUR").endswith(("Bcf", "MMcf"))
    assert any(m.label.startswith("Breakeven Henry Hub") for m in at.metric)


def test_analog_defaults_follow_the_analog():
    at = run("Horizontal oil (tight)", MODES[1], source="Example well")
    b = lambda: next(n.value for n in at.number_input if n.label == "b-factor")
    first = b()
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[3]).run()
    assert not at.exception
    assert b() != first


@pytest.mark.parametrize("deck", ["Escalating", "Custom deck"])
def test_price_decks(deck):
    at = run("Horizontal oil (tight)", MODES[0])
    flat_npv = metric(at, "NPV @ 10%")
    next(r for r in at.radio if r.label == "Forecast prices").set_value(deck).run()
    assert not at.exception
    if deck == "Escalating":  # higher prices later -> higher NPV than flat
        assert metric(at, "NPV @ 10%") != flat_npv
