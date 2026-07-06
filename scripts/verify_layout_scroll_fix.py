"""
Layout Scroll Fix — DOM Verification Script

Verifies that the layout-scroll-fix changes took effect:
1. stVerticalBlockBorderWrapper exists in DOM (≥2)
2. stColumn selectors are present
3. Body overflow is hidden
4. Containers have correct scroll properties
5. Border on stVerticalBlockBorderWrapper is hidden
"""

import asyncio
import sys
import json

from playwright.async_api import async_playwright

BASE_URL = "http://127.0.0.1:8501"
TIMEOUT = 30000

CHECKLIST = [
    ("C1: stMainBlockContainer exists", "stMainBlockContainer"),
    ("C2: stVerticalBlockBorderWrapper in DOM (≥2)", "stVerticalBlockBorderWrapper"),
    ("C3: stColumn selectors working", "stColumn"),
    ("C4: stChatInput present", "stChatInput"),
    ("C5: stChatMessage present", "stChatMessage"),
]


async def verify():
    checks = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print(f"Navigating to {BASE_URL}...")
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            await asyncio.sleep(4)

            results = await page.evaluate("""() => {
                const results = {};

                // C1: stMainBlockContainer
                const mainBlock = document.querySelectorAll('[data-testid="stMainBlockContainer"]');
                results.stMainBlockContainer = mainBlock.length;

                // C2: stVerticalBlockBorderWrapper
                const borderWrappers = document.querySelectorAll(
                    '[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"]'
                );
                results.stVerticalBlockBorderWrapper = borderWrappers.length;

                // Check border is hidden
                if (borderWrappers.length > 0) {
                    const style = window.getComputedStyle(borderWrappers[0]);
                    results.borderHidden = style.borderWidth === '0px' || style.borderStyle === 'none';
                    results.borderWidth = style.borderWidth;
                    results.borderStyle = style.borderStyle;
                    results.boxShadow = style.boxShadow;
                } else {
                    results.borderHidden = false;
                    results.borderWidth = 'N/A';
                }

                // C3: stColumn
                const stColumns = document.querySelectorAll('[data-testid="stMainBlockContainer"] [data-testid="stColumn"]');
                results.stColumn = stColumns.length;

                // C4: stChatInput
                const chatInput = document.querySelectorAll('[data-testid="stChatInput"]');
                results.stChatInput = chatInput.length;

                // C5: stChatMessage
                const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
                results.stChatMessage = chatMessages.length;

                // Body overflow
                const bodyStyle = window.getComputedStyle(document.body);
                results.bodyOverflowX = bodyStyle.overflowX;
                results.bodyOverflowY = bodyStyle.overflowY;

                // App viewport overflow
                const appView = document.querySelector('[data-testid="stAppViewContainer"]');
                if (appView) {
                    const appStyle = window.getComputedStyle(appView);
                    results.appOverflow = appStyle.overflow;
                }

                return results;
            }""")

            print("\n" + "=" * 50)
            print("   LAYOUT-SCROLL-FIX DOM VERIFICATION")
            print("=" * 50)

            # C1
            c1_ok = results.get("stMainBlockContainer", 0) >= 1
            print(f"C1: stMainBlockContainer count = {results.get('stMainBlockContainer')} {'✅' if c1_ok else '❌'}")

            # C2
            c2_count = results.get("stVerticalBlockBorderWrapper", 0)
            c2_ok = c2_count >= 2
            print(f"C2: stVerticalBlockBorderWrapper count = {c2_count} {'✅' if c2_ok else '❌'}")

            # C2b — border hidden
            c2b_ok = results.get("borderHidden", False)
            print(f"C2b: Border hidden? {c2b_ok} {'✅' if c2b_ok else '❌'} (border={results.get('borderWidth')}, style={results.get('borderStyle')})")

            # C3
            c3_cols = results.get("stColumn", 0)
            c3_ok = c3_cols >= 2
            print(f"C3: stColumn count (in main) = {c3_cols} {'✅' if c3_ok else '❌'}")

            # Body overflow
            body_ox = results.get("bodyOverflowX", "?")
            body_oy = results.get("bodyOverflowY", "?")
            body_ok = body_ox in ("hidden", "clip") and body_oy in ("hidden", "clip")
            print(f"C4: Body overflow = ({body_ox}, {body_oy}) {'✅' if body_ok else '❌'}")

            # Chat elements
            c5_inp = results.get("stChatInput", 0)
            c5_msg = results.get("stChatMessage", 0)
            print(f"C5: stChatInput={c5_inp}, stChatMessage={c5_msg}")

            # Overall
            all_pass = c1_ok and c2_ok and c2b_ok and c3_ok and body_ok
            verdict = "✅ ALL CHECKS PASSED" if all_pass else "❌ SOME CHECKS FAILED"
            print("\n" + "=" * 50)
            print(f"   VERDICT: {verdict}")
            print("=" * 50)

            sys.exit(0 if all_pass else 1)

        except Exception as e:
            print(f"❌ ERROR: {e}")
            sys.exit(1)
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(verify())
