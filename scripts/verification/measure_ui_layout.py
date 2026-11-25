"""Measure the rendered main UI layout at several viewport sizes.

Run against a live app at ``--url``. Writes one JSON file and one PNG per
viewport so a layout change can be compared against a real browser frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from playwright.async_api import ViewportSize, async_playwright

MEASURE_JS = r"""
() => {
    const rect = (el) => {
        if (!el) return null;
        const r = el.getBoundingClientRect();
        const cs = getComputedStyle(el);
        return {
            x: Math.round(r.x), y: Math.round(r.y),
            w: Math.round(r.width), h: Math.round(r.height),
            bottom: Math.round(r.bottom),
            clientHeight: el.clientHeight,
            scrollHeight: el.scrollHeight,
            overflowY: cs.overflowY,
            position: cs.position,
            fontSize: cs.fontSize,
        };
    };
    const q = (s, root = document) => root.querySelector(s);
    const main = q('[data-testid="stMainBlockContainer"]');
    const sidebar = q('[data-testid="stSidebar"]');
    const header = q('[data-testid="stHeader"]');
    const input = q('[data-testid="stChatInput"]');
    const inputCol = input?.closest('[data-testid="stColumn"]');
    const scroller = inputCol
        ? [...inputCol.querySelectorAll('*')].find(
            (el) => ['auto', 'scroll'].includes(getComputedStyle(el).overflowY)
          )
        : null;
    const row = q('[data-testid="stHorizontalBlock"]', main || document);
    const rowColumns = row
        ? [...row.children].filter(
            (el) => el.getAttribute('data-testid') === 'stColumn'
          )
        : [];
    const messages = [...document.querySelectorAll(
        '[data-testid="stChatMessage"]'
    )];
    const statuses = [...document.querySelectorAll('[data-testid="stStatusWidget"]')];
    const expanders = [...document.querySelectorAll('[data-testid="stExpander"]')];
    const brand = q('.rag-brand');
    return {
        viewport: {w: innerWidth, h: innerHeight},
        document: {scrollWidth: document.documentElement.scrollWidth},
        header: rect(header),
        sidebar: rect(sidebar),
        main: rect(main),
        brand: rect(brand),
        row: rect(row),
        rowColumns: rowColumns.map(rect),
        scroller: rect(scroller),
        input: rect(input),
        messages: messages.slice(0, 12).map(rect),
        messageCount: messages.length,
        statuses: statuses.map(rect),
        expanders: expanders.slice(0, 8).map(rect),
        statusText: statuses.map((el) => el.innerText.slice(0, 120)),
    };
}
"""

VIEWPORTS: dict[str, ViewportSize] = {
    "desktop": {"width": 1280, "height": 800},
    "laptop": {"width": 1024, "height": 768},
    "mobile": {"width": 390, "height": 844},
}


async def capture(
    url: str,
    out_dir: Path,
    viewport: ViewportSize,
    label: str,
    wait_s: float,
) -> dict[str, Any]:
    """Capture one browser frame and its measurements."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport=viewport)
        page = await context.new_page()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                break
            except Exception:
                await asyncio.sleep(2)
        await page.wait_for_selector(
            '[data-testid="stChatInput"]', timeout=int(wait_s * 1000)
        )
        await page.wait_for_timeout(1500)
        measured = await page.evaluate(MEASURE_JS)
        await page.screenshot(path=str(out_dir / f"{label}.png"))
        measured["label"] = label
        await browser.close()
        return measured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", default="reports/ui-layout-measure")
    parser.add_argument(
        "--viewports", nargs="*", default=list(VIEWPORTS), choices=list(VIEWPORTS)
    )
    parser.add_argument("--wait", type=float, default=60.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for label in args.viewports:
        result[label] = asyncio.run(
            capture(args.url, out_dir, VIEWPORTS[label], label, args.wait)
        )
    target = out_dir / "measure.json"
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for label, frame in result.items():
        columns = [f"{c['w']}px" for c in frame["rowColumns"]]
        print(
            f"{label}: viewport={frame['viewport']} columns={columns} "
            f"scroller={frame['scroller']} input={frame['input']}"
        )
    print(f"evidence: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
