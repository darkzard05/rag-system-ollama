import asyncio
from playwright.async_api import async_playwright

async def analyze_dom():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Navigating to app...")
            await page.goto("http://localhost:8501", wait_until="networkidle")
            await asyncio.sleep(3)
            
            # 1. 모든 data-testid 추출
            test_ids = await page.evaluate("""() => {
                const elements = document.querySelectorAll('[data-testid]');
                return Array.from(elements).map(el => ({
                    testid: el.getAttribute('data-testid'),
                    tagName: el.tagName,
                    className: el.className
                }));
            }""")
            
            print("\n--- All data-testids found ---")
            for item in test_ids:
                print(f"Tag: {item['tagName']}, ID: {item['testid']}, Class: {item['className']}")
            
            # 2. overflow-y가 auto/scroll인 요소 찾기
            scrollable = await page.evaluate("""() => {
                const all = document.querySelectorAll('*');
                const result = [];
                all.forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        result.push({
                            tagName: el.tagName,
                            className: el.className,
                            testid: el.getAttribute('data-testid'),
                            height: style.height
                        });
                    }
                });
                return result;
            }""")
            
            print("\n--- Scrollable Elements (overflow-y: auto/scroll) ---")
            for item in scrollable:
                print(f"Tag: {item['tagName']}, ID: {item['testid']}, Class: {item['className']}, Height: {item['height']}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(analyze_dom())
