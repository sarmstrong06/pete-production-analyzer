import re

import numpy as np
import pytest

from production.report import build_pdf, render_chart

DEPTH = np.arange(3100, 3200, 0.5)
CHARTS = [
    {"type": "log_tracks", "depth": DEPTH, "pay": (DEPTH > 3140) & (DEPTH < 3150),
     "tracks": [{"x": np.random.default_rng(0).uniform(20, 100, len(DEPTH)), "label": "GR"},
                {"x": np.full(len(DEPTH), 10.0), "label": "Rt", "logx": True}]},
    {"type": "xy", "logy": True, "xlabel": "Months", "ylabel": "bbl/day",
     "series": [{"x": [0, 1, 2], "y": [100, 80, 60], "label": "Actual", "style": "scatter"},
                {"x": [0, 1, 2, 3], "y": [100, 79, 61, 50], "label": "Fit", "style": "line", "color": "#eb6834"},
                {"x": [0, 3], "y": [90, 40], "style": "dash"}]},
    {"type": "bar", "x": [2026, 2027, 2028], "y": [-2.5, 1.2, 0.8], "ylabel": "$M"},
    {"type": "hist", "samples": np.random.default_rng(1).normal(1, 1, 500), "marks": {"P90": -0.3, "P50": 1, "P10": 2.3}},
    {"type": "tornado", "labels": ["Recovery factor", "Price"], "low": [-1.0, -0.5], "high": [1.5, 0.6]},
]


@pytest.mark.parametrize("spec", CHARTS, ids=[c["type"] for c in CHARTS])
def test_each_chart_renders_png(spec):
    assert render_chart(spec).getvalue()[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_chart_type():
    with pytest.raises(ValueError):
        render_chart({"type": "pie"})


def test_build_pdf_with_every_part():
    report = {
        "title": "Upstream appraisal: NUSS #1",
        "subtitle": "Vertical oil (conventional) · API 15-009-25378 · φ, Sw, → and ≤ render in Unicode",
        "generated": "Generated 2026-10-25",
        "headline": [("Net pay", "23.0 ft"), ("EUR", "203 Mbbl")],
        "verdict": ("review", "Cross-check: review. Plausible but outside the typical range."),
        "notes": ["A note."],
        "sections": [
            {"heading": "Well log", "text": ["Paragraph."],
             "tables": [{"title": "Inputs", "header": ["Input", "Value"], "rows": [["Zone", "3,106-3,342 ft"]]}],
             "charts": CHARTS[:2]},
            {"heading": "Economics", "new_page": True, "charts": CHARTS[2:]
             + [{"type": "row", "charts": [CHARTS[3], CHARTS[4]]}]},
        ],
        "sources": ["KGS", "RRC"],
        "footer": "Test footer",
    }
    pdf = build_pdf(report)
    assert pdf[:5] == b"%PDF-"
    assert len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf)) >= 2   # the second section starts a new page
    assert len(pdf) > 20_000   # the charts are embedded


def test_minimal_report():
    assert build_pdf({"title": "Only a title"})[:5] == b"%PDF-"
