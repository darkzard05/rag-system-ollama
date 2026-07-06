import asyncio
from playwright.async_api import async_playwright, expect


async def verify_layout():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        print("Connecting to app...")
        try:
            await page.goto("http://localhost:8501", timeout=30000)
        except Exception as e:
            print(f"Error: Could not connect to app. Is it running? {e}")
            return

        # Wait for the app to load - use chat input as a marker
        print("Waiting for app to load...")
        try:
            await page.wait_for_selector(
                '[data-testid="stChatInputContainer"]', timeout=15000
            )
        except Exception:
            print("Error: App did not load the chat input in time.")
            return

        # 1. Verify Global Scroll is disabled
        # Streamlit sometimes wraps things, let's check multiple levels
        body_overflow = await page.evaluate(
            "window.getComputedStyle(document.body).overflow"
        )
        html_overflow = await page.evaluate(
            "window.getComputedStyle(document.documentElement).overflow"
        )
        app_overflow = await page.evaluate(
            "window.getComputedStyle(document.querySelector('.stApp')).overflow"
        )

        print(
            f"Body overflow: {body_overflow}, HTML overflow: {html_overflow}, App overflow: {app_overflow}"
        )

        global_scroll_disabled = (
            body_overflow == "hidden"
            or html_overflow == "hidden"
            or app_overflow == "hidden"
        )
        print(f"Global scroll disabled: {global_scroll_disabled}")

        # 2. Verify Independent Column Scrolling
        columns = await page.query_selector_all('div[data-testid="column"]')
        print(f"Found {len(columns)} columns")
        if len(columns) < 2:
            print("Error: Less than 2 columns found.")
        else:
            pdf_col = columns[0]
            chat_col = columns[1]
            pdf_overflow = await pdf_col.evaluate(
                "el => window.getComputedStyle(el).overflowY"
            )
            chat_overflow = await chat_col.evaluate(
                "el => window.getComputedStyle(el).overflowY"
            )
            print(
                f"PDF Col overflowY: {pdf_overflow}, Chat Col overflowY: {chat_overflow}"
            )

        # 3. Verify Chat Input Visibility
        input_container = await page.query_selector(
            '[data-testid="stChatInputContainer"]'
        )
        if input_container:
            box = await input_container.bounding_box()
            viewport = page.viewport_size
            if box and viewport:
                is_visible = (
                    box["x"] >= 0
                    and box["y"] >= 0
                    and (box["x"] + box["width"]) <= viewport["width"]
                    and (box["y"] + box["height"]) <= viewport["height"]
                )
                print(
                    f"Chat input in viewport: {is_visible} (y: {box['y']}, height: {box['height']})"
                )
            else:
                print("Bounding box or viewport not found.")
        else:
            print("Chat input container not found.")

        # 4. Test Auto-scroll
        chat_input = await page.query_selector('textarea[data-testid="stChatInput"]')
        if chat_input:
            for i in range(5):
                await chat_input.fill(f"Test message {i}")
                await chat_input.press("Enter")
                await asyncio.sleep(2)

            is_at_bottom = await chat_col.evaluate("""
                el => {
                    const wrapper = el.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
                    if (!wrapper) return false;
                    return Math.abs(wrapper.scrollHeight - wrapper.scrollTop - wrapper.clientHeight) < 10;
                }
            """)
            print(f"Chat auto-scrolled to bottom: {is_at_bottom}")
        else:
            print("Chat input textarea not found.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(verify_layout())
