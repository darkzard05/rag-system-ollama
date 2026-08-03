"""
Layout Scroll Fix — DOM Verification Script

Verifies that the independent-column-scroll fix took effect (Streamlit 1.54 DOM):
1. stMainBlockContainer exists
2. Chat column stVerticalBlock is the single scroll container (overflow-y: auto)
3. Message wrappers are natural height (no per-message micro-scrollbars)
4. stChatInput is present (page-level bottom bar)
5. Body / app overflow is hidden (page lock)

NOTE: older checks for `stVerticalBlockBorderWrapper` were removed — that node
does not exist in Streamlit 1.54, so the previous script could never pass.
"""

import asyncio
import sys

from playwright.async_api import async_playwright

BASE_URL = "http://127.0.0.1:8501"
TIMEOUT = 30000

CHECKS = [
    ("C1: stMainBlockContainer exists", "stMainBlockContainer"),
    ("C2: stColumn selectors working", "stColumn"),
    ("C3: chat column stVerticalBlock is the scroller", "chatScroller"),
    (
        "C4: message wrappers are natural height (no micro-scrollbars)",
        "noMicroScrollbars",
    ),
    ("C5: stChatInput present", "stChatInput"),
    ("C6: body overflow hidden (page lock)", "pageLocked"),
]


async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print(f"Navigating to {BASE_URL}...")
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            await asyncio.sleep(4)

            results = await page.evaluate("""() => {
                const results = {};

                const mainBlock = document.querySelectorAll('[data-testid="stMainBlockContainer"]');
                results.stMainBlockContainer = mainBlock.length;

                const columns = document.querySelectorAll('[data-testid="stMainBlockContainer"] [data-testid="stColumn"]');
                results.stColumn = columns.length;

                const chatColumn = Array.from(columns).find(
                    (col) => col.querySelector('[data-testid="stChatMessage"]')
                );
                if (chatColumn) {
                    const vb = chatColumn.querySelector(':scope > [data-testid="stVerticalBlock"]');
                    results.chatScroller = vb
                        ? getComputedStyle(vb).overflowY
                        : 'NO_VB';
                    if (vb) {
                        const offending = Array.from(vb.children).filter((child) => {
                            const cs = getComputedStyle(child);
                            return cs.overflowY === 'auto' || cs.overflowY === 'scroll';
                        });
                        results.noMicroScrollbars = offending.length === 0;
                    } else {
                        results.noMicroScrollbars = false;
                    }
                } else {
                    results.chatScroller = 'NO_CHAT_COLUMN';
                    results.noMicroScrollbars = false;
                }

                results.stChatInput = document.querySelectorAll('[data-testid="stChatInput"]').length;

                const app = document.querySelector('.stApp');
                results.pageLocked = app ? getComputedStyle(app).overflow === 'hidden' : false;

                return results;
            }""")

            print("\n" + "=" * 50)
            print("   INDEPENDENT-SCROLL DOM VERIFICATION")
            print("=" * 50)

            c1 = results.get("stMainBlockContainer", 0) >= 1
            c2 = results.get("stColumn", 0) >= 2
            c3 = results.get("chatScroller") in ("auto", "scroll")
            c4 = results.get("noMicroScrollbars", False)
            c5 = results.get("stChatInput", 0) >= 1
            c6 = results.get("pageLocked", False)

            print(
                f"C1: stMainBlockContainer count = {results.get('stMainBlockContainer')} {'✅' if c1 else '❌'}"
            )
            print(
                f"C2: stColumn count = {results.get('stColumn')} {'✅' if c2 else '❌'}"
            )
            print(
                f"C3: chat scroller overflowY = {results.get('chatScroller')} {'✅' if c3 else '❌'}"
            )
            print(
                f"C4: no per-message micro-scrollbars = {results.get('noMicroScrollbars')} {'✅' if c4 else '❌'}"
            )
            print(
                f"C5: stChatInput count = {results.get('stChatInput')} {'✅' if c5 else '❌'}"
            )
            print(
                f"C6: page locked (.stApp overflow hidden) = {results.get('pageLocked')} {'✅' if c6 else '❌'}"
            )

            all_pass = c1 and c2 and c3 and c4 and c5 and c6
            verdict = "✅ ALL CHECKS PASSED" if all_pass else "❌ SOME CHECKS FAILED"
            print("=" * 50)
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
