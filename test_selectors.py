import asyncio

from playwright.async_api import async_playwright


async def test_style(page, selector):
    print(f"Testing selector: {selector}")
    # Inject style
    await page.add_style_tag(
        content=f"{selector} {{ position: fixed !important; bottom: 10px !important; left: 10px !important; background: red !important; z-index: 9999 !important; }}"
    )

    # Check if any element with this selector now has position: fixed
    elements = await page.locator(selector).all()
    if not elements:
        print("  Result: Selector not found")
        return False

    for el in elements:
        style = await el.evaluate("el => window.getComputedStyle(el).position")
        if style == "fixed":
            print("  Result: SUCCESS! Element is now fixed.")
            return True

    print("  Result: FAILED. Position is not fixed.")
    return False


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501")
        await page.wait_for_selector('[data-testid="stChatInput"]', timeout=10000)

        selectors = [
            '[data-testid="stChatInput"]',
            ".st-key-main_chat_input",
            'div[data-testid="stElementContainer"]:has([data-testid="stChatInput"])',
            '[data-testid="stChatInputContainer"]',
        ]

        for s in selectors:
            await test_style(page, s)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
