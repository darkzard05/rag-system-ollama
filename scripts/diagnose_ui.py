
import asyncio
from playwright.async_api import async_playwright

async def diagnose():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501", wait_until="networkidle")
        await asyncio.sleep(5)

        print("--- Page Title ---")
        print(await page.title())

        print("\n--- All elements with data-testid ---")
        elements = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('[data-testid]')).map(el => ({
                testid: el.getAttribute('data-testid'),
                tagName: el.tagName,
                className: el.className
            }));
        }""")
        for el in elements:
            print(f"{el['tagName']} {el['className']} -> {el['testid']}")

        print("\n--- Body Style ---")
        print(await page.evaluate("window.getComputedStyle(document.body).overflow"))

        await browser.close()

if __name__ == "__main__":
    asyncio.run(diagnose())
