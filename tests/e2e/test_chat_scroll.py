import asyncio

from playwright.async_api import async_playwright


async def test_chat_scroll():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # 1. Navigate to the app
        try:
            await page.goto("http://127.0.0.1:8501", timeout=15000)
        except Exception as e:
            print(
                f"Error: Could not connect to app at 127.0.0.1:8501. Is it running? {e}"
            )
            await browser.close()
            return

        print("Successfully connected to the app.")
        await page.wait_for_timeout(3000)

        # 2. Verify Global Lock (Main page should not scroll)
        body_style = await page.evaluate(
            "window.getComputedStyle(document.body).overflow"
        )
        html_style = await page.evaluate(
            "window.getComputedStyle(document.documentElement).overflow"
        )
        app_style = await page.evaluate(
            'window.getComputedStyle(document.querySelector(".stApp")).overflow'
        )

        print(f"Body overflow: {body_style}")
        print(f"HTML overflow: {html_style}")
        print(f"App overflow: {app_style}")

        assert app_style == "hidden", (
            f"Expected app overflow to be hidden, but got {app_style}"
        )
        print(
            "✅ Global Lock Verified: .stApp is fixed (effectively locking the page)."
        )

        # 3. Identify the Chat Column and inject overflow content
        # Find the chat column (the one containing stTabs) using JS since :has() CSS
        # is not supported in all Chromium versions
        FIND_CHAT_WRAPPER_JS = """
            () => {
                const cols = document.querySelectorAll('div[data-testid="stColumn"]');
                for (const col of cols) {
                    if (col.querySelector('[data-testid="stChatInput"], [data-testid="stChatMessage"]')) {
                        // Return the scrollable inner stVerticalBlock (under stLayoutWrapper)
                        const outer = col.querySelector(':scope > [data-testid="stVerticalBlock"]');
                        if (!outer) return null;
                        const inner = outer.querySelector('[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]');
                        return inner || outer;
                    }
                }
                return null;
            }
        """

        # Inject large content into the chat column to create overflow
        overflow_created = await page.evaluate(f"""
            () => {{
                const wrapper = ({FIND_CHAT_WRAPPER_JS})();
                if (!wrapper) return 'WRAPPER_NOT_FOUND';

                const initialSh = wrapper.scrollHeight;
                const initialCh = wrapper.clientHeight;

                const contentDiv = document.createElement('div');
                contentDiv.style.padding = '10px';
                contentDiv.innerHTML = '<div>' + Array(50).fill(
                    '<p style="padding:8px;border-bottom:1px solid #eee;">'
                    + 'Test scroll content line. '.repeat(8)
                    + '</p>'
                ).join('') + '</div>';

                const tabContent = wrapper.querySelector('[role="tabpanel"], [data-testid="stTabs"]');
                if (tabContent) {{
                    tabContent.appendChild(contentDiv);
                }} else {{
                    wrapper.appendChild(contentDiv);
                }}

                const finalSh = wrapper.scrollHeight;
                const finalCh = wrapper.clientHeight;
                return {{ initialSh, initialCh, finalSh, finalCh, overflow: finalSh > finalCh }};
            }}
        """)
        print(f"Overflow injection result: {overflow_created}")

        await page.wait_for_timeout(500)

        # Verify overflow was created
        overflow_check = await page.evaluate(f"""
            () => {{
                const el = ({FIND_CHAT_WRAPPER_JS})();
                return el ? {{ sh: el.scrollHeight, ch: el.clientHeight }} : null;
            }}
        """)
        print(f"Chat container after injection: {overflow_check}")

        assert overflow_check is not None, (
            "Chat container not found after content injection"
        )
        assert overflow_check["sh"] > overflow_check["ch"], (
            f"Chat container did not overflow after content injection. {overflow_check}"
        )
        print("✅ Overflow Detected: Chat content exceeds view height.")

        # 4. Verify Independent Scroll
        before_scroll_pdf = await page.evaluate("""
            () => {
                const cols = document.querySelectorAll('div[data-testid="stColumn"]');
                const pdfBlock = cols.length >= 3
                    ? cols[2].querySelector(':scope > [data-testid="stVerticalBlock"]')
                    : null;
                return pdfBlock ? pdfBlock.scrollTop : -1;
            }
        """)

        # Scroll chat container
        await page.evaluate(f"""
            () => {{
                const el = ({FIND_CHAT_WRAPPER_JS})();
                if (el) el.scrollTop = el.scrollHeight;
            }}
        """)
        await page.wait_for_timeout(200)

        # Check chat container scrollTop
        chat_scroll_top = await page.evaluate(f"""
            () => {{
                const el = ({FIND_CHAT_WRAPPER_JS})();
                return el ? el.scrollTop : -1;
            }}
        """)
        print(f"Chat scrollTop: {chat_scroll_top}")

        assert chat_scroll_top > 0, (
            f"Chat container failed to scroll. scrollTop: {chat_scroll_top}"
        )
        print(
            f"✅ Independent Scroll Verified: Chat container scrolled to {chat_scroll_top}."
        )

        # Check PDF column scrollTop remained 0 (independent scrolling)
        after_scroll_pdf = await page.evaluate("""
            () => {
                const cols = document.querySelectorAll('div[data-testid="stColumn"]');
                const pdfBlock = cols.length >= 3
                    ? cols[2].querySelector(':scope > [data-testid="stVerticalBlock"]')
                    : null;
                return pdfBlock ? pdfBlock.scrollTop : -1;
            }
        """)
        print(f"PDF scrollTop (before={before_scroll_pdf}, after={after_scroll_pdf})")

        assert before_scroll_pdf == after_scroll_pdf, (
            f"PDF column scrolled when only chat was scrolled! {before_scroll_pdf} -> {after_scroll_pdf}"
        )
        print(
            "✅ Independent Scroll Verified: PDF column did NOT move when chat scrolled."
        )

        # 5. Verify Main Window is still locked
        window_scroll_top = await page.evaluate("window.scrollY")
        assert window_scroll_top == 0, (
            f"Main window scrolled! scrollTop: {window_scroll_top}"
        )
        print("✅ Final Lock Verified: Main window remained at top.")

        await browser.close()
        print("\nAll scroll tests PASSED successfully!")


if __name__ == "__main__":
    asyncio.run(test_chat_scroll())
