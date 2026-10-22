"""E2E: first upload frame mounts the viewer and a compact analysis card.

The upload callback publishes the first-page preview and a PREPARING
``BuildState`` before the rerun. The browser should therefore see:

1. the PDF iframe, and
2. a collapsed analysis card (PREPARING) rather than an expanded stepper.

This test records the first observed state with a MutationObserver so a short
PREPARING frame cannot be missed by polling.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.async_api import Page, async_playwright

BASE_URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")
PDF_REL_PATH = "tests/data/2201.07520v1.pdf"

INSTALL_RECORDER_JS = """
(name) => {
    const visible = (el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    };
    window.__uploadProbe = {
        name,
        t0: performance.now(),
        firstFrame: null,
        firstPanel: null,
        panelOpen: null,
        panelText: "",
        done: false,
    };
    const probe = window.__uploadProbe;
    const check = () => {
        if (probe.firstFrame === null) {
            const frame = Array.from(document.querySelectorAll('iframe')).find(visible);
            if (frame) probe.firstFrame = performance.now() - probe.t0;
        }
        if (probe.firstPanel === null) {
            const panel = Array.from(document.querySelectorAll('details')).find(
                (d) => (d.innerText || '').includes(name)
            );
            if (panel) {
                probe.firstPanel = performance.now() - probe.t0;
                probe.panelOpen = panel.open;
                probe.panelText = (panel.innerText || '').slice(0, 240);
            }
        }
        if (!probe.done && probe.firstFrame !== null && probe.firstPanel !== null) {
            probe.done = true;
            observer.disconnect();
        }
    };
    const observer = new MutationObserver(check);
    observer.observe(document.body, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true,
    });
    check();
}
"""


async def _open_app(page: Page) -> bool:
    try:
        await page.goto(BASE_URL, timeout=20000)
        return True
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_first_upload_frame_has_viewer_and_compact_panel():
    filename = Path(PDF_REL_PATH).name
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1366, "height": 768})
        if not await _open_app(page):
            await browser.close()
            pytest.skip(f"Streamlit app not running at {BASE_URL}")

        await page.wait_for_timeout(1500)
        await page.evaluate(INSTALL_RECORDER_JS, filename)
        await page.set_input_files(
            '[data-testid="stSidebar"] input[type="file"]', PDF_REL_PATH
        )
        try:
            await page.wait_for_function(
                "() => window.__uploadProbe && window.__uploadProbe.done",
                timeout=30000,
            )
            probe = await page.evaluate("() => window.__uploadProbe")
        finally:
            await browser.close()

    assert probe["firstFrame"] is not None, "PDF first page iframe never mounted"
    assert probe["firstPanel"] is not None, "analysis card never mounted"
    assert probe["panelOpen"] is False, (
        f"PREPARING card must start collapsed, got open={probe['panelOpen']} "
        f"text={probe['panelText']!r}"
    )
    assert probe["firstPanel"] - probe["firstFrame"] < 5000, (
        "analysis card must mount in the same first-frame window as the viewer "
        f"(frame={probe['firstFrame']:.0f}ms panel={probe['firstPanel']:.0f}ms)"
    )
    print(f"first frame probe: {probe}")
