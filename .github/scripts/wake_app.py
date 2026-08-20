"""
Visits the live dashboard with a real (headless) browser so Streamlit
Community Cloud sees an actual viewer session, not just a static HTTP hit.

A plain curl/requests ping only ever fetches the pre-JS HTML shell -- it
can't run the JS bundle that opens the WebSocket connection to the Python
backend, which is what Streamlit Cloud actually uses to decide whether an
app has a live viewer. That's why curl-based pings kept reporting success
while the app kept going to sleep anyway.
"""

import sys

from playwright.sync_api import sync_playwright

URL = "https://saas-sentiment-analyzer.streamlit.app/"
DASHBOARD_MARKER = "Product Insights Dashboard"
WAKE_BUTTON_TEXT = "get this app back up"


def get_app_frame(page):
    # Streamlit Community Cloud renders the actual app inside a nested
    # iframe (path contains "/~/+/"), separate from the outer shell page
    # and an unrelated statuspage.io embed also present on the page.
    for frame in page.frames:
        if "/~/+/" in frame.url:
            return frame
    return None


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle", timeout=60_000)

        frame = get_app_frame(page)
        if frame is None:
            raise RuntimeError("Could not find the app's embedded frame.")

        wake_button = frame.get_by_text(WAKE_BUTTON_TEXT, exact=False)
        if wake_button.count() > 0:
            print("App was asleep -- clicking wake button.")
            wake_button.first.click()
            page.wait_for_timeout(5_000)
            frame = get_app_frame(page) or frame  # re-acquire if it reloaded

        frame.wait_for_selector(f"text={DASHBOARD_MARKER}", timeout=90_000)
        print("Dashboard rendered successfully -- app is awake.")
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed to confirm the app is awake: {e}")
        sys.exit(1)
