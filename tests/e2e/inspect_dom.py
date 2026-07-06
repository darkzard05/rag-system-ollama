import asyncio
from playwright.async_api import async_playwright


async def inspect():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501")

        # Send a message to populate the UI
        input_selector = 'div[data-testid="stChatInput"] textarea'
        try:
            await page.fill(input_selector, "Hello! Please generate a long response.")
            await page.keyboard.press("Enter")
            await asyncio.sleep(5)  # Wait for response
        except Exception as e:
            print(f"Interaction failed: {e}")

        # Get all elements with data-testid
        test_ids = await page.evaluate("""
            () => {
                const elements = Array.from(document.querySelectorAll('[data-testid]'));
                return elements.map(el => ({
                    testid: el.getAttribute('data-testid'),
                    tagName: el.tagName,
                    className: el.className,
                    height: el.clientHeight,
                    scrollHeight: el.scrollHeight
                }));
            }
        """)
        for item in test_ids:
            print(item)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(inspect())
