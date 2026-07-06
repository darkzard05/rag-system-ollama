import asyncio
from playwright.async_api import async_playwright

async def dump_html():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Navigating to app...")
            await page.goto("http://localhost:8501", wait_until="networkidle")
            await asyncio.sleep(3) # Stabilize
            
            content = await page.content()
            with open("dom_dump.html", "w", encoding="utf-8") as f:
                f.write(content)
            print("HTML dumped to dom_dump.html")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(dump_html())
