
import asyncio
from playwright.async_api import async_playwright

async def diagnose_input():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501", wait_until="networkidle")
        await asyncio.sleep(5)

        input_el = page.locator('[data-testid="stChatInput"]')
        if await input_el.count() > 0:
            style = await input_el.evaluate("el => window.getComputedStyle(el).position")
            bottom = await input_el.evaluate("el => window.getComputedStyle(el).bottom")
            left = await input_el.evaluate("el => window.getComputedStyle(el).left")
            print(f"stChatInput: position={style}, bottom={bottom}, left={left}")
        else:
            print("stChatInput not found")

        # Check for any element that might be the container
        containers = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('div')).filter(el => 
                el.className.includes('stChatInput') || el.getAttribute('data-testid')?.includes('ChatInput')
            ).map(el => ({
                testid: el.getAttribute('data-testid'),
                className: el.className,
                position: window.getComputedStyle(el).position
            }));
        }""")
        print("\nPotential containers:")
        for c in containers:
            print(c)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(diagnose_input())
