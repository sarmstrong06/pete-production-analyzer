"""PDF appraisal report.

The dashboard describes the report as plain data (a dict): headline results,
a verdict, and sections of paragraphs, tables and chart specs. This module
lays it out with fpdf2 and draws the charts with matplotlib, so it has no
Streamlit dependency and can be tested on its own.

Chart specs (dicts with a "type"):
    xy         series: list of {x, y, label, color, style ("line"|"dash"|"dot"|"scatter"), width}
               xlabel, ylabel, logy, title
    bar        x, y, xlabel, ylabel, title
    hist       samples, marks {label: value}, xlabel, title
    tornado    labels, low, high (changes from base), xlabel, title
    log_tracks depth, tracks: list of {x, label, logx}, pay (bool array), title
    row        charts: two of the above (not log_tracks), drawn side by side
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402
from fpdf.fonts import FontFace  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"
INK, SOFT = (33, 37, 41), (110, 110, 110)
VERDICT_COLORS = {"pass": (27, 175, 122), "review": (237, 161, 0), "fail": (214, 69, 65), "info": (42, 120, 214)}
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu ships with matplotlib (full Unicode)
CHART_SIZE = {"log_tracks": (7.2, 4.0), "xy": (7.2, 2.9)}  # inches; the PDF scales them to the page width
HALF_SIZE = (4.2, 3.4)  # one chart of a side-by-side "row"


# ---------------------------------------------------------------- charts
def _style(ax, title=None):
    ax.grid(True, color="#d9d8d6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b0aea9")
    ax.tick_params(colors="#555555", labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color="#222222", fontweight="bold")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_chart(spec):
    """PNG (BytesIO) of one chart spec."""
    kind = spec["type"]
    if kind == "log_tracks":
        tracks = spec["tracks"]
        fig, axes = plt.subplots(1, len(tracks), figsize=CHART_SIZE["log_tracks"], sharey=True)
        depth, pay = np.asarray(spec["depth"], float), np.asarray(spec.get("pay", []), bool)
        step = float(np.median(np.diff(depth))) if len(depth) > 1 else 0.5
        for ax, tr in zip(np.atleast_1d(axes), tracks):
            ax.plot(np.asarray(tr["x"], float), depth, color=BLUE, linewidth=0.9)
            if tr.get("logx"):
                ax.set_xscale("log")
            if pay.any():
                runs = np.split(np.flatnonzero(pay), np.flatnonzero(np.diff(np.flatnonzero(pay)) > 1) + 1)
                for r in runs:
                    ax.axhspan(depth[r[0]] - step / 2, depth[r[-1]] + step / 2, color=AQUA, alpha=0.25, linewidth=0)
            _style(ax)
            ax.set_title(tr["label"], fontsize=9, color="#333333")
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Depth (ft)", fontsize=8)
        if spec.get("title"):
            fig.suptitle(spec["title"], fontsize=10, x=0.02, ha="left", fontweight="bold")
        return _png(fig)

    fig, ax = plt.subplots(figsize=spec.get("size", CHART_SIZE["xy"]))
    if kind == "xy":
        for s in spec["series"]:
            style = s.get("style", "line")
            if style == "scatter":
                ax.scatter(s["x"], s["y"], s=s.get("size", 10), color=s.get("color", BLUE), label=s.get("label"),
                           zorder=3, alpha=s.get("alpha", 1.0), edgecolors="white", linewidths=0.3)
            else:
                ls = {"line": "-", "dash": "--", "dot": ":"}[style]
                ax.plot(s["x"], s["y"], ls, color=s.get("color", BLUE), linewidth=s.get("width", 1.8),
                        label=s.get("label"), alpha=s.get("alpha", 1.0))
        if spec.get("logy"):
            ax.set_yscale("log")
        if any(s.get("label") for s in spec["series"]):
            ax.legend(fontsize=7, frameon=False, loc="best")
    elif kind == "bar":
        y = np.asarray(spec["y"], float)
        ax.bar(spec["x"], y, color=[BLUE if v >= 0 else ORANGE for v in y])
        ax.axhline(0, color=MUTED, linewidth=0.8)
    elif kind == "hist":
        ax.hist(spec["samples"], bins=50, color=BLUE, alpha=0.85)
        for label, v in spec.get("marks", {}).items():
            ax.axvline(v, color=MUTED, linestyle="-" if label == "P50" else "--", linewidth=1.2)
            ax.annotate(label, (v, 0.98), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7,
                        color="#444444", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_ylabel("Simulations", fontsize=8)
    elif kind == "tornado":
        labels = list(spec["labels"])[::-1]
        low, high = np.asarray(spec["low"], float)[::-1], np.asarray(spec["high"], float)[::-1]
        ax.barh(labels, low, color=BLUE, label="Input at low")
        ax.barh(labels, high, color=ORANGE, label="Input at high")
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.tick_params(axis="y", labelsize=7)
    else:
        raise ValueError(f"Unknown chart type {kind!r}")
    ax.set_xlabel(spec.get("xlabel", ""), fontsize=8)
    if kind != "hist":
        ax.set_ylabel(spec.get("ylabel", ""), fontsize=8)
    _style(ax, spec.get("title"))
    return _png(fig)


# ---------------------------------------------------------------- PDF
class _Report(FPDF):
    footer_text = ""

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(*SOFT)
        self.cell(0, 6, f"{self.footer_text}   ·   page {self.page_no()} of {{nb}}", align="C")


def build_pdf(r):
    """PDF bytes for a report dict (see module docstring and app.py for the layout)."""
    pdf = _Report(format="Letter")
    pdf.footer_text = r.get("footer", "PETE Production Analyzer")
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    bottom = pdf.h - 18  # lowest y for content, above the footer

    def text(s, size=9.5, bold=False, color=INK, h=5):
        pdf.set_font("DejaVu", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(width, h, s, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

    def room(mm):
        """Start a new page unless mm of vertical space is left."""
        if pdf.get_y() + mm > bottom:
            pdf.add_page()

    def table(t):
        rows = ([t["header"]] if t.get("header") else []) + [[str(c) for c in row] for row in t["rows"]]
        room(len(rows) * 5.6 + (7 if t.get("title") else 0) + 2)  # keep a table and its title together
        if t.get("title"):
            text(t["title"], size=10, bold=True, h=6)
        pdf.set_font("DejaVu", "", 8.5)
        pdf.set_text_color(*INK)
        pdf.set_fill_color(255, 255, 255)
        with pdf.table(col_widths=t.get("widths"), first_row_as_headings=bool(t.get("header")),
                       line_height=5.2, text_align=t.get("align", "LEFT"), borders_layout="HORIZONTAL_LINES",
                       headings_style=FontFace(emphasis="BOLD", fill_color=(241, 240, 237))) as tb:
            for row in rows:
                tr = tb.row()
                for c in row:
                    tr.cell(str(c))
        pdf.ln(2)

    def chart(spec):
        if spec["type"] == "row":  # two charts side by side
            imgs = [render_chart({**c, "size": HALF_SIZE}) for c in spec["charts"]]
            half = (width - 4) / 2
            h_mm = half * HALF_SIZE[1] / HALF_SIZE[0]
            room(h_mm + 2)
            y = pdf.get_y()
            for i, img in enumerate(imgs):
                pdf.image(img, x=pdf.l_margin + i * (half + 4), y=y, w=half, h=h_mm, keep_aspect_ratio=True)
            pdf.set_y(y + h_mm + 2)
            return
        img = render_chart(spec)
        h_mm = width * (CHART_SIZE["log_tracks"][1] / 7.2 if spec["type"] == "log_tracks" else CHART_SIZE["xy"][1] / 7.2)
        room(h_mm + 2)
        pdf.image(img, x=pdf.l_margin, w=width)
        pdf.ln(2)

    # Title block
    text(r["title"], size=18, bold=True, h=9)
    if r.get("subtitle"):
        text(r["subtitle"], size=10, color=SOFT)
    if r.get("generated"):
        text(r["generated"], size=8, color=SOFT)
    pdf.ln(3)

    if r.get("headline"):
        text("Key results", size=12, bold=True, h=7)
        table({"rows": r["headline"], "widths": (55, 45)})

    if r.get("verdict"):
        level, msg = r["verdict"]
        pdf.set_fill_color(*VERDICT_COLORS.get(level, VERDICT_COLORS["info"]))
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("DejaVu", "B", 9.5)
        pdf.multi_cell(width, 6, msg, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT, padding=2, align="L")
        pdf.set_fill_color(255, 255, 255)
        pdf.ln(3)

    for note in r.get("notes", []):
        text(note, size=9)
        pdf.ln(1)

    for sec in r.get("sections", []):
        if sec.get("new_page"):
            pdf.add_page()
        else:
            pdf.ln(3)
            room(45)  # don't leave a heading alone at the bottom of a page
        text(sec["heading"], size=13, bold=True, h=8)
        for para in sec.get("text", []):
            text(para, size=9)
            pdf.ln(1)
        for t in sec.get("tables", []):
            table(t)
        for spec in sec.get("charts", []):
            chart(spec)

    if r.get("sources"):
        pdf.ln(3)
        text("Data sources and notes", size=10, bold=True, h=6)
        for s in r["sources"]:
            text(f"• {s}", size=8, color=SOFT, h=4.2)
    return bytes(pdf.output())
