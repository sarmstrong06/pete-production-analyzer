"""Capture the README screenshots from a running dashboard (the Nuss 1 example).

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
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=120_000)
    page.wait_for_timeout(ms)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1180}, device_scale_factor=1.5,
                                color_scheme="light")
        page.goto(URL)
        page.get_by_text("Upstream appraisal").first.wait_for(timeout=120_000)
        settle(page)
        page.get_by_text("Real example: Nuss 1, Kansas").click()
        settle(page)
        page.get_by_text("Producing well: its own production").click()
        settle(page)
        page.get_by_role("button", name="Look up production").click()
        try:
            page.get_by_text("Kansas reports oil sales").wait_for(timeout=120_000)
        except Exception:
            page.screenshot(path=OUT / "debug_failed_lookup.png", full_page=True)
            raise
        settle(page, 2500)
        page.screenshot(path=OUT / "summary.png")

        for tab, name in [("1 · Well log", "well_log"), ("3 · Production forecast", "forecast"),
                          ("4 · Economics", "economics")]:
            page.get_by_role("tab", name=tab).click()
            settle(page, 2500)
            page.screenshot(path=OUT / f"{name}.png")
        browser.close()
    print("Saved", sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
