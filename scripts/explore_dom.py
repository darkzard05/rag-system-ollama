"""
Explore DOM structure - dump all data-testid attributes and border-related elements.
"""
import asyncio
from playwright.async_api import async_playwright

async def explore():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501", wait_until="networkidle")
        await asyncio.sleep(4)

        # Dump ALL data-testid attributes in the main area
        result = await page.evaluate("""() => {
            const all = document.querySelectorAll("[data-testid]");
            const testids = {};
            all.forEach(el => {
                const tid = el.getAttribute("data-testid");
                if (!testids[tid]) testids[tid] = 0;
                testids[tid]++;
            });
            return testids;
        }""")

        print("All data-testid attributes and counts:")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")

        # Check border-related elements
        border_related = await page.evaluate("""() => {
            const all = document.querySelectorAll("[data-testid*='Border'], [data-testid*='border']");
            return Array.from(all).map(el => ({
                testid: el.getAttribute("data-testid"),
                tag: el.tagName,
                classes: el.className
            }));
        }""")

        print()
        print("Border-related elements:")
        for el in border_related:
            print(f"  {el}")

        await browser.close()

asyncio.run(explore())
