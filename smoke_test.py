import asyncio

from playwright.async_api import async_playwright


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="C:\\Users\\darkzard05\\AppData\\Local\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe",
            headless=True,
        )
        page = await browser.new_page()
        try:
            await page.goto("http://localhost:8501", timeout=60000)
            await page.screenshot(path="smoke_test_screenshot.png", full_page=True)
            print("Screenshot saved as smoke_test_screenshot.png")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
