import asyncio
from playwright.async_api import async_playwright

async def check_css():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000)
        
        styles = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll("style"))
                    .map(s => s.innerHTML)
                    .filter(text => text.includes(".st-key-main_chat_input"));
            }
        """)
        print(f"Found {len(styles)} style tags containing .st-key-main_chat_input")
        for i, s in enumerate(styles):
            print(f"Style {i}:\n{s}")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(check_css())
