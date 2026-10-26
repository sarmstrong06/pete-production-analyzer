"""Capture the README screenshots from a running dashboard.

The screenshots show only public data: the Nuss 1 log (KCC, via the KGS)
appraised as a prospect, and production from the Texas Railroad Commission.
They don't show Kansas monthly production, which the KGS licenses for research
use, not redistribution.

Needs Playwright (pip install playwright && playwright install chromium) and the
dashboard running:  streamlit run app.py
Then:  python docs/take_screenshots.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).parent / "screenshots"


def settle(page, ms=1500):
    """Wait for Streamlit to finish rerunning."""
    page.wait_for_timeout(500)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=180_000)
    page.wait_for_timeout(ms)


def click(page, text):
    page.get_by_text(text, exact=True).first.click()
    settle(page)


def tab(page, name, path, scroll=0):
    page.get_by_role("tab", name=name).click()
    settle(page, 2500)
    if scroll:
        page.mouse.move(900, 600)
        page.mouse.wheel(0, scroll)
        settle(page, 800)
    page.screenshot(path=path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1180}, device_scale_factor=1.5,
                                color_scheme="light")
        page.goto(URL)
        page.get_by_text("Upstream appraisal").first.wait_for(timeout=120_000)
        settle(page)

        # The real Nuss 1 log appraised as a prospect (vertical oil, typical assumptions)
        click(page, "Real example: Nuss 1, Kansas")
        settle(page, 2500)
        page.screenshot(path=OUT / "summary.png")
        tab(page, "1 · Well log", OUT / "well_log.png")
        tab(page, "4 · Economics", OUT / "economics.png")
        tab(page, "5 · Uncertainty", OUT / "uncertainty.png", scroll=520)

        # Decline curves fitted to a Spraberry lease (Texas RRC data)
        click(page, "Horizontal oil (tight)")
        click(page, "Producing well: its own production")
        click(page, "Example well")
        # The cleanest-fitting example lease
        page.locator('[data-testid="stSelectbox"]').filter(has_text="Spraberry oil lease").locator("input").click()
        page.keyboard.type("ACT A (50678)")
        page.keyboard.press("Enter")
        settle(page)
        tab(page, "3 · Production forecast", OUT / "forecast.png")

        # A horizontal tight-oil prospect shaped like the bundled Spraberry wells
        click(page, "Prospect: field type curve")
        tab(page, "3 · Production forecast", OUT / "type_curve.png")
        browser.close()
    print("Saved", sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
