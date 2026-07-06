import asyncio
from playwright.async_api import async_playwright

async def analyze_paths():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Navigating to app...")
            await page.goto("http://localhost:8501", wait_until="networkidle")
            await asyncio.sleep(3)
            
            # Find all stVerticalBlock elements and trace their parent chain
            results = await page.evaluate("""() => {
                const blocks = document.querySelectorAll('[data-testid="stVerticalBlock"]');
                return Array.from(blocks).map(block => {
                    const path = [];
                    let current = block;
                    while (current && current.tagName !== 'BODY') {
                        path.push({
                            tagName: current.tagName,
                            testid: current.getAttribute('data-testid'),
                            className: current.className
                        });
                        current = current.parentElement;
                    }
                    return {
                        target: {
                            tagName: block.tagName,
                            testid: block.getAttribute('data-testid'),
                            className: block.className
                        },
                        path: path
                    };
                });
            }""")
            
            print("\n--- stVerticalBlock Parent Chains ---")
            for i, res in enumerate(results):
                print(f"\nBlock {i+1}:")
                for step in reversed(res['path']):
                    testid = step['testid'] if step['testid'] else "None"
                    print(f"  -> {step['tagName']}.{testid} ({step['className'][:50]}...)")
                print(f"  => TARGET: {res['target']['tagName']}.{res['target']['testid']}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(analyze_paths())
