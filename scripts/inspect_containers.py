import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to http://localhost:8501...")
        await page.goto("http://localhost:8501", wait_until="networkidle")
        
        # Wait a bit for Streamlit to fully render
        await asyncio.sleep(5)

        print("\n--- All elements with class names ---")
        classes = await page.evaluate("""() => {
            const elements = Array.from(document.querySelectorAll('*'));
            const classMap = {};
            elements.forEach(el => {
                if (el.className && typeof el.className === 'string') {
                    const classes = el.className.split(/\s+/).filter(c => c.length > 0);
                    classes.forEach(c => {
                        if (!classMap[c]) classMap[c] = [];
                        classMap[c].push(el.tagName.toLowerCase());
                    });
                }
            });
            return classMap;
        }""")
        import json
        print(json.dumps(classes, indent=2))

        print("\n--- All elements with data-testid ---")
        testids = await page.evaluate("""() => {
            const elements = Array.from(document.querySelectorAll('*'));
            const testidMap = {};
            elements.forEach(el => {
                const tid = el.getAttribute('data-testid');
                if (tid) {
                    if (!testidMap[tid]) testidMap[tid] = [];
                    testidMap[tid].push(el.tagName.toLowerCase());
                }
            });
            return testidMap;
        }""")
        print(json.dumps(testids, indent=2))

        print("\n--- Full Body HTML (first 5000 chars) ---")
        body_html = await page.evaluate("document.body.outerHTML")
        print(body_html[:5000])

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
