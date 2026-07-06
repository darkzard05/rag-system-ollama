import asyncio
from playwright.async_api import async_playwright

async def verify_styles_directly():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Navigating to app...")
            await page.goto("http://localhost:8501", wait_until="networkidle")
            await asyncio.sleep(3)
            
            # Find all stVerticalBlocks and check their styles
            results = await page.evaluate("""() => {
                const blocks = document.querySelectorAll('[data-testid="stVerticalBlock"]');
                return Array.from(blocks).map((el, idx) => {
                    const style = window.getComputedStyle(el);
                    return {
                        index: idx,
                        overflowY: style.overflowY,
                        height: style.height,
                        className: el.className,
                        parentTestId: el.parentElement ? el.parentElement.getAttribute('data-testid') : 'None'
                    };
                });
            }""")
            
            print("\n--- stVerticalBlock Style Audit ---")
            found_scrollable = False
            for item in results:
                status = "✅" if item['overflowY'] in ['auto', 'scroll'] else "❌"
                print(f"Block {item['index']}: {status} overflowY={item['overflowY']}, height={item['height']}, parent={item['parentTestId']}")
                if item['overflowY'] in ['auto', 'scroll']:
                    found_scrollable = True
            
            if not found_scrollable:
                print("\nCRITICAL: No stVerticalBlock has overflow-y: auto/scroll. CSS NOT APPLIED.")
            else:
                print("\nSUCCESS: Found scrollable stVerticalBlock(s)!")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(verify_styles_directly())
