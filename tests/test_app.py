"""Run the dashboard headlessly in each appraisal mode with the bundled example
data (no network) and check it renders without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MODES = ("Prospect: typical assumptions", "Prospect: decline from an analog well",
         "Producing well: its own production")
WELL_TYPES = ("Vertical (conventional)", "Horizontal (tight oil)")


def run(well_type, mode):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[1].set_value(well_type).run()
    at.sidebar.radio[2].set_value(mode).run()
    return at


@pytest.mark.parametrize("well_type", WELL_TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_dashboard_renders(well_type, mode):
    at = run(well_type, mode)
    assert not at.exception, [e.value for e in at.exception]
    metrics = {m.label for m in at.metric}
    assert {"Oil in place", "EUR", "NPV @ 10%"} <= metrics


def test_analog_defaults_follow_the_analog():
    at = run("Horizontal (tight oil)", MODES[1])
    b = lambda: next(n.value for n in at.number_input if n.label == "b-factor")
    first = b()
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[3]).run()
    assert not at.exception
    assert b() != first
