import asyncio
import json
import time
from playwright.async_api import async_playwright

PDF = r"C:\Users\DARKZA~1\AppData\Local\Temp\opencode\test_upload.pdf"
URL = "http://localhost:8501"


def snap(page):
    return page.evaluate(
        """() => {
        const main = document.querySelector('[data-testid="stMainBlockContainer"]');
        const cols = [...document.querySelectorAll('[data-testid="stColumn"]')].map(c => { const r=c.getBoundingClientRect(); return {h: Math.round(r.height), top: Math.round(r.top)}; });
        const hb = document.querySelector('[data-testid="stHorizontalBlock"]');
        const hbr = hb ? hb.getBoundingClientRect() : null;
        // main content stLayoutWrapper (the one whose parent chain hits stMainBlockContainer)
        const allw = [...document.querySelectorAll('[data-testid="stLayoutWrapper"]')];
        const mainw = allw.find(w => { let e=w.parentElement; while(e){ if(e.getAttribute('data-testid')==='stMainBlockContainer') return true; e=e.parentElement;} return false; });
        const mwr = mainw ? mainw.getBoundingClientRect() : null;
        return {
          main_h: main?Math.round(main.getBoundingClientRect().height):null,
          main_content_wrapper_h: mwr?Math.round(mwr.height):null,
          hblock: hbr?{h:Math.round(hbr.height),top:Math.round(hbr.top)}:null,
          cols,
          status: document.querySelectorAll('[data-testid="stStatusWidget"]').length,
        };
    }"""
    )


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_selector('[data-testid="stFileUploader"]', timeout=30000)
        await asyncio.sleep(3)
        head_css = await page.evaluate(
            """() => {
                const el = window.parent ? window.parent.document.getElementById('main-app-css') : null;
                const el2 = document.getElementById('main-app-css');
                return {parent_head: !!el, local_head: !!el2};
            }"""
        )
        print("HEAD_CSS", json.dumps(head_css, ensure_ascii=False))
        # Upload
        await page.locator(
            '[data-testid="stFileUploader"] input[type="file"]'
        ).set_input_files(PDF)
        # Rapid poll to catch the flash
        for i in range(10):
            await asyncio.sleep(0.12)
            s = await snap(page)
            print(f"t+{((i + 1) * 0.12):.2f}s", json.dumps(s, ensure_ascii=False))
        await asyncio.sleep(7)
        s = await snap(page)
        print("done", json.dumps(s, ensure_ascii=False))
        await browser.close()


asyncio.run(main())
