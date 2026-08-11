"""Verify the st.html-injected header-height script actually sets the --header-h CSS variable.

Modeled on scripts/dump_dom.py. Checks, against the RUNNING app (default localhost:8501):

1. Precondition: the injected script (T7 st.html migration) is present in the served page.
2. `[data-testid="stHeader"]` exists in the parent frame.
3. `document.documentElement.style.getPropertyValue('--header-h')` resolves to a non-empty
   `px`-suffixed value after the iframe script has had time to run.

Exit code 0 = PASS (variable set), 1 = FAIL (null/empty => st.html sandbox blocks parent access),
2 = QA failure (app not reachable / GOTO error).

Usage: python scripts/verify_header_height.py [--url http://localhost:8501]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from playwright.async_api import Page, async_playwright

SCRIPT_MARKER = "--header-h"


async def check_precondition(page: Page) -> tuple[bool, str]:
    """Return (found, evidence) for the injected script being present in the served page."""
    content = await page.content()
    found = SCRIPT_MARKER in content
    snippet = ""
    if found:
        idx = content.find(SCRIPT_MARKER)
        snippet = content[max(0, idx - 80) : idx + 120].replace("\n", " ")
    return found, snippet


async def verify(url: str) -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print(f"[1/4] Navigating to {url} ...")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)  # Let the Streamlit script + iframe load.

            # --- Precondition: injected script present in served page ---
            found, snippet = await check_precondition(page)
            print(
                f"[2/4] Precondition: injected script marker '{SCRIPT_MARKER}' in page.content() -> {found}"
            )
            if snippet:
                print(f"      context: ...{snippet}...")

            # stHeader presence
            header_count = await page.locator('[data-testid="stHeader"]').count()
            print(f"[3/4] [data-testid='stHeader'] count -> {header_count}")

            # --- Poll for the CSS variable (iframe script needs a beat to run) ---
            value: str | None = None
            for _ in range(15):
                value = await page.evaluate(
                    "document.documentElement.style.getPropertyValue('--header-h')"
                )
                if value and value.strip():
                    break
                await asyncio.sleep(1)
            value = (value or "").strip()
            print(f"[4/4] getPropertyValue('--header-h') -> {value!r}")

            if header_count == 0:
                print(
                    "RESULT: FAIL (stHeader missing — page did not render Streamlit header)"
                )
                return 1
            if not value:
                print(
                    "RESULT: FAIL (--header-h empty — st.html sandbox blocks parent-frame "
                    "access, or script did not execute)"
                )
                return 1
            if not value.endswith("px"):
                print(f"RESULT: FAIL (--header-h={value!r} does not end with 'px')")
                return 1
            print(
                f"RESULT: PASS (--header-h={value!r}, stHeader present, script executed)"
            )
            return 0
        except Exception as e:  # noqa: BLE001 - report any GOTO/browser failure as QA failure
            print(f"ERROR: {type(e).__name__}: {e}")
            return 2
        finally:
            await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    return asyncio.run(verify(args.url))


if __name__ == "__main__":
    sys.exit(main())
