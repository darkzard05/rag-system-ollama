"""
Integrated E2E Verification Script

Runs ALL layout verification in ONE browser session:
- DOM structure verification (replaces verify_dom_structure.py)
- Flex chain CSS property verification (replaces verify_ui_scrolling.py)
- Independent scrolling test (replaces test_chat_scroll.py)

Usage:
    1. Start Streamlit: streamlit run src/main.py
    2. Run: python scripts/verify_e2e_all.py

Expected time: ~15-20 seconds (vs 60-90s running 3 separate scripts)
"""

import asyncio
import datetime
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Configuration
STREAMLIT_PORT = os.environ.get("STREAMLIT_PORT", "8501")
BASE_URL = f"http://127.0.0.1:{STREAMLIT_PORT}"
EVIDENCE_FILE = Path(".omo/evidence/task-6-e2e-verification.txt")
TIMEOUT = 15000  # 15 seconds for page load


async def main():
    async with async_playwright() as p:
        print(f"Launching browser and navigating to {BASE_URL}...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ FAIL: Could not connect to app at {BASE_URL}: {e}")
            await browser.close()
            sys.exit(1)

        results = {"passed": 0, "failed": 0, "details": []}
        start_time = datetime.datetime.now()

        # === CHECK 1: DOM Structure ===
        print("Running Check 1: DOM Structure...")
        dom = await page.evaluate("""() => {
            const main = document.querySelector('[data-testid="stMainBlockContainer"]');
            const header = document.querySelector('[data-testid="stHeader"]');
            return {
                headerH: header ? header.getBoundingClientRect().height : null,
                mainExists: !!main,
                mainDisplay: main ? window.getComputedStyle(main).display : null,
                mainHeight: main ? window.getComputedStyle(main).height : null,
                windowH: window.innerHeight
            };
        }""")

        # Verify main height is approx window.innerHeight - 60
        expected_main_h = (dom["windowH"] or 0) - 60
        actual_main_h = float(dom["mainHeight"].replace('px', '')) if dom["mainHeight"] else 0
        height_ok = abs(actual_main_h - expected_main_h) <= 5

        c1_pass = (
            dom["headerH"] is not None
            and 50 <= dom["headerH"] <= 70
            and dom["mainExists"]
            and dom["mainDisplay"] == "block"
            and height_ok
        )

        c1_res = (
            f"=== CHECK 1: DOM Structure ===\n"
            f"Header height: {dom['headerH']}px {'✓' if 50 <= (dom['headerH'] or 0) <= 70 else '✗'} (expected ~60px)\n"
            f"stMainBlockContainer display: {dom['mainDisplay']} {'✓' if dom['mainDisplay'] == 'block' else '✗'}\n"
            f"stMainBlockContainer height: {dom['mainHeight']} {'✓' if height_ok else '✗'} (expected ~{expected_main_h}px)\n"
            f"Result: {'PASS' if c1_pass else 'FAIL'}\n"
        )
        results["details"].append(c1_res)
        if c1_pass:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # === CHECK 2: Flex Chain CSS Properties ===
        print("Running Check 2: Flex Chain CSS...")
        flex = await page.evaluate("""() => {
            const cols = document.querySelectorAll('[data-testid="stColumn"]');
            return Array.from(cols).map(c => {
                const s = window.getComputedStyle(c);
                const vb = c.querySelector(':scope > [data-testid="stVerticalBlock"]');
                const vs = vb ? window.getComputedStyle(vb) : null;
                return {
                    display: s.display,
                    overflowY: vs ? vs.overflowY : null,
                };
            });
        }""")

        verified_cols = [
            c for c in flex if c["display"] == "flex" and c["overflowY"] in ["auto", "scroll"]
        ]
        c2_pass = len(verified_cols) >= 2

        col_details = "\n".join(
            [
                f"  Col {i}: display={c['display']}, overflowY={c['overflowY']} {'✓' if c['display'] == 'flex' and c['overflowY'] in ['auto', 'scroll'] else '✗'}"
                for i, c in enumerate(flex)
            ]
        )

        c2_res = (
            f"=== CHECK 2: Flex Chain CSS ===\n"
            f"Found {len(flex)} columns:\n"
            f"{col_details}\n"
            f"Result: {'PASS' if c2_pass else 'FAIL'}\n"
        )
        results["details"].append(c2_res)
        if c2_pass:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # === CHECK 3: Independent Scrolling ===
        print("Running Check 3: Independent Scrolling...")

        # Helper to find chat wrapper
        FIND_CHAT_WRAPPER_JS = """
            () => {
                const mainContainer = document.querySelector('[data-testid="stMainBlockContainer"]');
                if (!mainContainer) return null;
                const cols = mainContainer.querySelectorAll('[data-testid="stColumn"]');
                for (const col of cols) {
                    if (col.querySelector('[data-testid="stChatInput"], [data-testid="stChatMessage"]')) {
                        return col.querySelector(':scope > [data-testid="stVerticalBlock"]');
                    }
                }
                return null;
            }
        """

        # Inject content
        overflow_data = await page.evaluate(f"""
            () => {{
                const wrapper = ({FIND_CHAT_WRAPPER_JS})();
                if (!wrapper) return null;
                const initialSh = wrapper.scrollHeight;
                const initialCh = wrapper.clientHeight;
                const contentDiv = document.createElement('div');
                contentDiv.style.padding = '10px';
                contentDiv.innerHTML = '<div>' + Array(50).fill(
                    '<p style="padding:8px;border-bottom:1px solid #eee;">'
                    + 'Test scroll content line. '.repeat(8)
                    + '</p>'
                ).join('') + '</div>';
                wrapper.appendChild(contentDiv);
                return {{ initialSh, initialCh, finalSh: wrapper.scrollHeight, finalCh: wrapper.clientHeight }};
            }}
        """)

        if not overflow_data:
            c3_res = "=== CHECK 3: Independent Scrolling ===\nChat wrapper not found\nResult: FAIL\n"
            c3_pass = False
        else:
            # Scroll chat
            await page.evaluate(f"""
                () => {{
                    const el = ({FIND_CHAT_WRAPPER_JS})();
                    if (el) el.scrollTop = el.scrollHeight;
                }}
            """)
            await asyncio.sleep(0.2)

            # Verify
            scroll_metrics = await page.evaluate(f"""
                () => {{
                    const chat = ({FIND_CHAT_WRAPPER_JS})();
                    const mainContainer = document.querySelector('[data-testid="stMainBlockContainer"]');
                    const cols = mainContainer ? mainContainer.querySelectorAll('[data-testid="stColumn"]') : [];
                    const pdf = cols.length >= 1 ? cols[0].querySelector(':scope > [data-testid="stVerticalBlock"]') : null;
                    return {{
                        chatScrollTop: chat ? chat.scrollTop : -1,
                        pdfScrollTop: pdf ? pdf.scrollTop : -1,
                        windowScrollY: window.scrollY
                    }};
                }}
            """)

            overflow_ok = overflow_data["finalSh"] > overflow_data["finalCh"]
            chat_scrolled = scroll_metrics["chatScrollTop"] > 0
            pdf_unmoved = scroll_metrics["pdfScrollTop"] == 0
            window_locked = scroll_metrics["windowScrollY"] == 0

            c3_pass = overflow_ok and chat_scrolled and pdf_unmoved and window_locked

            c3_res = (
                f"=== CHECK 3: Independent Scrolling ===\n"
                f"Overflow created: scrollHeight({overflow_data['finalSh']}) > clientHeight({overflow_data['finalCh']}) {'✓' if overflow_ok else '✗'}\n"
                f"Chat column scrollTop after scroll: {scroll_metrics['chatScrollTop']} {'✓' if chat_scrolled else '✗'}\n"
                f"PDF column after scroll: {scroll_metrics['pdfScrollTop']} {'✓' if pdf_unmoved else '✗'} (unchanged)\n"
                f"Main window scrollY: {scroll_metrics['windowScrollY']} {'✓' if window_locked else '✗'}\n"
                f"Result: {'PASS' if c3_pass else 'FAIL'}\n"
            )

        results["details"].append(c3_res)
        if c3_pass:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # === FINAL RESULTS ===
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = (
            f"=== E2E VERIFICATION RESULTS ===\n"
            f"Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            + "\n".join(results["details"])
            + f"\n=== SUMMARY ===\n"
            f"All 3 checks: {'PASS' if results['failed'] == 0 else 'FAIL'}\n"
            f"Duration: {duration:.1f}s\n"
        )

        print(summary)

        # Save to evidence file
        EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
            f.write(summary)

        await browser.close()
        sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
