import os

import pytest
from playwright.async_api import async_playwright

BASE_URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")


FIND_CHAT_SCROLLER_JS = """
    () => {
        const mainContainer = document.querySelector('[data-testid="stMainBlockContainer"]');
        if (!mainContainer) return null;
        const cols = mainContainer.querySelectorAll('[data-testid="stColumn"]');
        for (const col of cols) {
            if (col.querySelector('[data-testid="stChatMessage"]')) {
                return col.querySelector(':scope > [data-testid="stVerticalBlock"]');
            }
        }
        return null;
    }
"""

FIND_PDF_SCROLLER_JS = """
    () => {
        const mainContainer = document.querySelector('[data-testid="stMainBlockContainer"]');
        if (!mainContainer) return null;
        const col = mainContainer.querySelector('[data-testid="stColumn"]');
        if (!col) return null;
        const stack = [col];
        while (stack.length) {
            const el = stack.pop();
            if (el !== col) {
                const cs = getComputedStyle(el);
                if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {
                    return el;
                }
            }
            stack.push(...el.children);
        }
        return null;
    }
"""

CHECK_INPUT_PINNED_JS = f"""
    () => {{
        const scroller = ({FIND_CHAT_SCROLLER_JS})();
        if (!scroller) return null;
        const input = scroller.querySelector('[data-testid="stChatInput"]');
        if (!input) return null;
        const wrapper = input.parentElement;
        const sb = scroller.getBoundingClientRect();
        const wb = wrapper.getBoundingClientRect();
        return {{
            wrapperPosition: getComputedStyle(wrapper).position,
            inputPosition: getComputedStyle(input).position,
            delta: wb.bottom - sb.bottom,
            visible: wb.top < sb.bottom - 1 && wb.bottom > sb.top + 1,
        }};
    }}
"""


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_chat_scroll():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(BASE_URL, timeout=15000)
        except Exception as e:
            await browser.close()
            pytest.skip(f"Streamlit app not running at {BASE_URL} (e2e skipped): {e}")

        print("Successfully connected to the app.")
        await page.wait_for_selector('[data-testid="stChatInput"]', timeout=60000)
        # settle: let the chat column's scroll container (stColumn > stVerticalBlock)
        # and fragment-driven layout finish rendering before asserting geometry.
        await page.wait_for_timeout(2000)

        # 1. Global Lock (main window must not scroll)
        app_style = await page.evaluate(
            'window.getComputedStyle(document.querySelector(".stApp")).overflow'
        )
        assert app_style == "hidden", (
            f"Expected app overflow to be hidden, but got {app_style}"
        )
        print("✅ Global Lock Verified: .stApp overflow is hidden.")

        # 2. Chat column stVerticalBlock is the single scroll container
        chat_metrics = await page.evaluate(f"""
            () => {{
                const scroller = ({FIND_CHAT_SCROLLER_JS})();
                if (!scroller) return null;
                const cs = getComputedStyle(scroller);
                const children = Array.from(scroller.children).map((child) => {{
                    const ccs = getComputedStyle(child);
                    return {{
                        tag: child.getAttribute('data-testid'),
                        overflowY: ccs.overflowY,
                        flexGrow: ccs.flexGrow,
                        flexShrink: ccs.flexShrink,
                    }};
                }});
                return {{
                    overflowY: cs.overflowY,
                    childCount: children.length,
                    children,
                }};
            }}
        """)
        assert chat_metrics is not None, "Chat scroller not found"
        assert chat_metrics["overflowY"] in ("auto", "scroll"), (
            f"Chat column stVerticalBlock must be the scroller, got overflowY="
            f"{chat_metrics['overflowY']}"
        )
        print(
            f"✅ Chat column stVerticalBlock is the scroller "
            f"(overflowY={chat_metrics['overflowY']}, {chat_metrics['childCount']} children)."
        )

        # 3. No per-message micro-scrollbars: message wrappers must NOT scroll
        offending = [
            c
            for c in chat_metrics["children"]
            if c["overflowY"] in ("auto", "scroll") or c["flexGrow"] != "0"
        ]
        assert not offending, (
            f"Message wrappers must not be individual scrollers or flex:1 "
            f"({offending[:3]})"
        )
        print(
            "✅ Message wrappers are natural height (flex:0 0 auto, no micro-scrollbars)."
        )

        # 4. Chat input pinned at the column bottom (short content)
        pinned_short = await page.evaluate(f"(() => ({CHECK_INPUT_PINNED_JS})())()")
        assert pinned_short is not None, "Chat input not found"
        assert pinned_short["wrapperPosition"] == "sticky", (
            f"Input wrapper must be sticky, got {pinned_short['wrapperPosition']}"
        )
        assert pinned_short["inputPosition"] == "static", (
            f"Input itself must be static, got {pinned_short['inputPosition']}"
        )
        assert abs(pinned_short["delta"]) <= 2, (
            f"Input must sit at the column bottom, delta={pinned_short['delta']}"
        )
        assert pinned_short["visible"], f"Input must be visible, got {pinned_short}"
        print(f"✅ Input pinned at column bottom (short content): {pinned_short}")

        # 5. Create overflow in the chat scroller and verify it scrolls as one unit
        overflow_created = await page.evaluate(f"""
            () => {{
                const scroller = ({FIND_CHAT_SCROLLER_JS})();
                if (!scroller) return null;
                const contentDiv = document.createElement('div');
                contentDiv.style.padding = '10px';
                contentDiv.innerHTML = Array(50).fill(
                    '<p style="padding:8px;border-bottom:1px solid #eee;">'
                    + 'Test scroll content line. '.repeat(8)
                    + '</p>'
                ).join('');
                const inputWrapper = scroller.querySelector(
                    ':scope > [data-testid="stElementContainer"]:has(> [data-testid="stChatInput"])'
                );
                // Real messages are always appended BEFORE the input (which stays the last
                // element of the chat column), so inject in the same position.
                if (inputWrapper) {{
                    scroller.insertBefore(contentDiv, inputWrapper);
                }} else {{
                    scroller.appendChild(contentDiv);
                }}
                return {{
                    sh: scroller.scrollHeight,
                    ch: scroller.clientHeight,
                }};
            }}
        """)
        assert overflow_created is not None, (
            "Chat scroller not found for overflow injection"
        )
        assert overflow_created["sh"] > overflow_created["ch"], (
            f"Chat scroller did not overflow after injection: {overflow_created}"
        )
        print(f"✅ Overflow Detected in chat scroller: {overflow_created}")

        before_pdf = await page.evaluate(
            f"(() => {{ const el = ({FIND_PDF_SCROLLER_JS})(); return el ? el.scrollTop : null; }})()"
        )

        # 6. Scroll to bottom: input stays pinned over the content
        await page.evaluate(f"""
            () => {{
                const scroller = ({FIND_CHAT_SCROLLER_JS})();
                if (scroller) scroller.scrollTop = scroller.scrollHeight;
            }}
        """)
        await page.wait_for_timeout(200)

        chat_scroll_top = await page.evaluate(f"""
            () => {{
                const scroller = ({FIND_CHAT_SCROLLER_JS})();
                return scroller ? scroller.scrollTop : -1;
            }}
        """)
        assert chat_scroll_top > 0, (
            f"Chat scroller failed to scroll. scrollTop: {chat_scroll_top}"
        )
        print(f"✅ Chat scroller scrolled to {chat_scroll_top}.")

        pinned_bottom = await page.evaluate(f"(() => ({CHECK_INPUT_PINNED_JS})())()")
        assert pinned_bottom is not None, "Chat input not found"
        assert abs(pinned_bottom["delta"]) <= 2, (
            f"Input must stay pinned at the column bottom while scrolled, got {pinned_bottom}"
        )
        assert pinned_bottom["visible"], (
            f"Input must stay visible while scrolled, got {pinned_bottom}"
        )
        print(f"✅ Input pinned while scrolled to bottom: {pinned_bottom}")

        # 7. Scroll back to top (mid-scroll): input still pinned
        await page.evaluate(f"""
            () => {{
                const scroller = ({FIND_CHAT_SCROLLER_JS})();
                if (scroller) scroller.scrollTop = 0;
            }}
        """)
        await page.wait_for_timeout(200)
        pinned_top = await page.evaluate(f"(() => ({CHECK_INPUT_PINNED_JS})())()")
        assert pinned_top is not None, "Chat input not found"
        assert abs(pinned_top["delta"]) <= 2, (
            f"Input must stay pinned at the column bottom mid-scroll, got {pinned_top}"
        )
        assert pinned_top["visible"], (
            f"Input must stay visible mid-scroll, got {pinned_top}"
        )
        print(f"✅ Input pinned mid-scroll (scrollTop=0): {pinned_top}")

        # 8. PDF column must not move (independent scrolling)
        after_pdf = await page.evaluate(
            f"(() => {{ const el = ({FIND_PDF_SCROLLER_JS})(); return el ? el.scrollTop : null; }})()"
        )
        assert before_pdf == after_pdf, (
            f"PDF column scrolled when only chat was scrolled! {before_pdf} -> {after_pdf}"
        )
        print(f"✅ PDF column independent: scrollTop {before_pdf} -> {after_pdf}.")

        # 9. Main window still locked
        window_scroll_top = await page.evaluate("window.scrollY")
        assert window_scroll_top == 0, (
            f"Main window scrolled! scrollTop: {window_scroll_top}"
        )
        print("✅ Main window remained locked (scrollY=0).")

        await browser.close()
        print("\nAll independent-scroll tests PASSED successfully!")
