import asyncio
from playwright.async_api import async_playwright

async def debug_layout():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000)
        
        # 1. Print .stApp children structure
        structure = await page.evaluate("""
            () => {
                const app = document.querySelector(".stApp");
                if (!app) return "stApp not found";
                return Array.from(app.children).map(child => ({
                    tagName: child.tagName,
                    className: child.className,
                    height: child.clientHeight
                }));
            }
        """)
        print("--- .stApp Structure ---")
        print(structure)
        
        # 2. Print Chat Input detailed style
        input_style = await page.evaluate("""
            () => {
                const input = document.querySelector('.st-key-main_chat_input');
                if (!input) return "Input not found";
                const style = window.getComputedStyle(input);
                return {
                    position: style.position,
                    bottom: style.bottom,
                    top: style.top,
                    zIndex: style.zIndex,
                    display: style.display
                };
            }
        """)
        print("\n--- Chat Input Style ---")
        print(input_style)
        
        # 3. Check for any style tags that might be injecting the fix
        styles = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('style'))
                    .map(s => s.innerHTML)
                    .filter(text => text.includes('.st-key-main_chat_input'))
                    .join('\\n--- style tag ---\\n');
            }
        """)
        print("\n--- Injected Styles ---")
        print(styles)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_layout())
