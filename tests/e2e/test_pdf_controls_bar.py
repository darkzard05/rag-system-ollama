"""
E2E tests for the PDF controls bar (RED baseline for work plan `pdf-control-bar-fix`).

These tests capture the CURRENT (buggy) geometry of the PDF navigation bar so that
fixing it later turns them GREEN:

- The bar wrapper is sticky but currently rests ~64px above the column bottom.
- The "prev" button currently sits ~32px above the Page / "next" controls.

Both tests are expected to FAIL against the current app (RED). Do not weaken them.
"""

import os

import pytest
from playwright.async_api import Page, async_playwright

BASE_URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")

PDF_REL_PATH = r"tests/data/2201.07520v1.pdf"

# Locates the PDF controls bar wrapper (the sticky container that wraps the
# horizontal block holding prev / page / next controls).
FIND_BAR_WRAPPER_JS = """
    () => {
        let wrapper = document.querySelector(
            '[data-testid="stColumn"]:first-child '
            + '[data-testid="stLayoutWrapper"]:has(> [data-testid="stHorizontalBlock"])'
        );
        if (wrapper) return wrapper;
        const btn = Array.from(document.querySelectorAll('button')).find(
            (b) => b.textContent.includes('이전')
        );
        if (!btn) return null;
        const block = btn.closest('[data-testid="stHorizontalBlock"]');
        return block ? block.parentElement : null;
    }
"""

# Visible "이전" button present and laid out (a plain wait_for_selector can match a
# hidden node during the async upload rerun).
BAR_VISIBLE_JS = """
    () => {
        return Array.from(document.querySelectorAll('button')).some((b) => {
            if (!b.textContent.includes('이전')) return false;
            const r = b.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        });
    }
"""

BAR_METRICS_JS = f"""
    () => {{
        const col = document.querySelector('[data-testid="stColumn"]');
        if (!col) return null;
        const wrapper = ({FIND_BAR_WRAPPER_JS})();
        if (!wrapper) return null;
        const wr = wrapper.getBoundingClientRect();
        const cbr = col.getBoundingClientRect();
        const row = wrapper.querySelector('[data-testid="stHorizontalBlock"]');
        const topOf = (el) => {{
            if (!el) return null;
            const container = el.closest('[data-testid="stElementContainer"]');
            return container
                ? container.getBoundingClientRect().top
                : el.getBoundingClientRect().top;
        }};
        const prevBtn = Array.from(wrapper.querySelectorAll('button')).find(
            (b) => b.textContent.includes('이전')
        );
        const nextBtn = Array.from(wrapper.querySelectorAll('button')).find(
            (b) => b.textContent.includes('다음')
        );
        const pageInput = wrapper.querySelector('input[aria-label="Page"]');
        const tops = [topOf(prevBtn), topOf(pageInput), topOf(nextBtn)].filter(
            (t) => t !== null
        );
        return {{
            position: getComputedStyle(wrapper).position,
            barTop: wr.top,
            barBottom: wr.bottom,
            barHeight: wr.height,
            columnBottom: cbr.bottom,
            columnHeight: cbr.height,
            deltaBottom: wr.bottom - cbr.bottom,
            rowScrollWidth: row ? row.scrollWidth : null,
            rowClientWidth: row ? row.clientWidth : null,
            controlTops: tops,
            controlSpread:
                tops.length >= 2 ? Math.max(...tops) - Math.min(...tops) : null,
        }};
    }}
"""


async def _upload_and_settle(page: Page) -> None:
    """Upload the sample PDF through the sidebar and wait for the controls bar."""
    await page.set_input_files(
        '[data-testid="stSidebar"] input[type="file"]', PDF_REL_PATH
    )
    # Poll until a VISIBLE "이전" button exists (survives the async upload rerun).
    await page.wait_for_function(BAR_VISIBLE_JS, timeout=30000)
    # Let the pdf iframe settle its geometry after the rerun.
    await page.wait_for_timeout(1500)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_controls_bar_pinned_and_one_line_desktop():
    """Desktop (1366x768): bar is sticky, flush with the column bottom, one line."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1366, "height": 768})

        try:
            await page.goto(BASE_URL, timeout=15000)
        except Exception as e:
            await browser.close()
            pytest.skip(f"Streamlit app not running at {BASE_URL} (e2e skipped): {e}")

        await _upload_and_settle(page)
        metrics = await page.evaluate(f"(() => ({BAR_METRICS_JS})())()")
        await browser.close()

    assert metrics is not None, "PDF controls bar wrapper not found"
    assert metrics["position"] == "sticky", (
        f"Bar wrapper must be position:sticky, got {metrics['position']}"
    )
    assert abs(metrics["deltaBottom"]) <= 4, (
        f"Bar wrapper must sit flush with the column bottom, "
        f"deltaBottom={metrics['deltaBottom']:.1f}px (expected |delta| <= 4)"
    )
    assert metrics["controlSpread"] is not None, "Could not locate all three controls"
    assert metrics["controlSpread"] < 8, (
        f"Prev/Page/next controls must sit on one line, "
        f"controlSpread={metrics['controlSpread']:.1f}px (expected < 8), "
        f"tops={metrics['controlTops']}"
    )
    assert metrics["rowClientWidth"] is not None, "Bar row not found"
    assert metrics["rowScrollWidth"] <= metrics["rowClientWidth"], (
        f"Bar row must not overflow, scrollWidth={metrics['rowScrollWidth']} > "
        f"clientWidth={metrics['rowClientWidth']}"
    )
    print(f"✅ Desktop: bar metrics {metrics}")


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_controls_bar_sticky_in_narrow_viewport():
    """Narrow (700x900): bar stays sticky, flush with the column, no inflated height."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 700, "height": 900})

        try:
            await page.goto(BASE_URL, timeout=15000)
        except Exception as e:
            await browser.close()
            pytest.skip(f"Streamlit app not running at {BASE_URL} (e2e skipped): {e}")

        await _upload_and_settle(page)
        metrics = await page.evaluate(f"(() => ({BAR_METRICS_JS})())()")
        await browser.close()

    assert metrics is not None, "PDF controls bar wrapper not found"
    assert metrics["position"] == "sticky", (
        f"Bar wrapper must be position:sticky, got {metrics['position']}"
    )
    assert abs(metrics["deltaBottom"]) <= 4, (
        f"Bar wrapper must sit flush with the column bottom, "
        f"deltaBottom={metrics['deltaBottom']:.1f}px (expected |delta| <= 4)"
    )
    assert metrics["barHeight"] < 0.30 * metrics["columnHeight"], (
        f"Bar must not be inflated, barHeight={metrics['barHeight']:.1f}px "
        f"(expected < 0.30 * columnHeight={0.30 * metrics['columnHeight']:.1f}px)"
    )
    assert metrics["controlSpread"] is not None, "Could not locate all three controls"
    assert metrics["controlSpread"] < 8, (
        f"Prev/Page/next controls must sit on one line, "
        f"controlSpread={metrics['controlSpread']:.1f}px (expected < 8), "
        f"tops={metrics['controlTops']}"
    )
    print(f"✅ Narrow: bar metrics {metrics}")
