"""Inspect how the running app on :8501 renders st.html (or not)."""

import asyncio

from playwright.async_api import async_playwright


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(
            "http://localhost:8501", wait_until="networkidle", timeout=30000
        )
        await asyncio.sleep(6)
        print(
            "stHtml count:",
            await page.locator(
                "div[data-testid=stHtml], iframe[data-testid=stHtml]"
            ).count(),
        )
        testids = await page.locator("[data-testid]").evaluate_all(
            "(els) => els.map(e => e.getAttribute('data-testid')).filter(t => t && t.toLowerCase().includes('html'))"
        )
        print("testids containing 'html':", testids)
        iframe_count = await page.locator("iframe").count()
        print("iframe count:", iframe_count)
        for i in range(iframe_count):
            src = await page.locator("iframe").nth(i).get_attribute("src")
            srcdoc = await page.locator("iframe").nth(i).get_attribute("srcdoc")
            cls = await page.locator("iframe").nth(i).get_attribute("class")
            print(
                f"iframe[{i}] src={str(src)[:60]!r} srcdoc={str(srcdoc)[:60]!r} class={cls!r}"
            )
        txt = await page.locator("body").inner_text()
        print("has_exception_text:", ("Exception" in txt) or ("Traceback" in txt))
        content = await page.content()
        for marker in ["--header-h", "stHeader", "3.75rem"]:
            print(f"marker {marker!r} in page.content():", marker in content)
        await browser.close()


asyncio.run(main())
