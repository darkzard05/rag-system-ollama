import asyncio
import json
import time
from playwright.async_api import async_playwright

PDF = r"C:\Users\DARKZA~1\AppData\Local\Temp\opencode\test_upload.pdf"
URL = "http://localhost:8501"


def snapshot_struct(page):
    """Capture the 2-column container nesting + key heights."""
    return page.evaluate(
        """() => {
        const out = {};
        const main = document.querySelector('[data-testid="stMainBlockContainer"]');
        out.main_h = main ? main.getBoundingClientRect().height : null;
        // All stLayoutWrapper elements with their ancestor chain testids
        const wrappers = [...document.querySelectorAll('[data-testid="stLayoutWrapper"]')];
        out.layout_wrappers = wrappers.map(w => {
            const chain = [];
            let el = w;
            for (let i=0; i<6 && el; i++) {
                chain.push(el.getAttribute('data-testid'));
                el = el.parentElement;
            }
            const r = w.getBoundingClientRect();
            return {chain, h: Math.round(r.height), top: Math.round(r.top)};
        });
        const hb = document.querySelector('[data-testid="stHorizontalBlock"]');
        if (hb) { const r = hb.getBoundingClientRect(); out.hblock = {h: Math.round(r.height), top: Math.round(r.top)}; }
        const cols = [...document.querySelectorAll('[data-testid="stColumn"]')];
        out.cols = cols.map(c => { const r=c.getBoundingClientRect(); return {h: Math.round(r.height), top: Math.round(r.top)}; });
        // count fragment wrappers / extra vertical blocks directly under main
        const mainVb = main ? [...main.children].map(c => c.getAttribute('data-testid')) : [];
        out.main_children = mainVb;
        // status / expander presence (analysis panel)
        out.has_status = !!document.querySelector('[data-testid="stStatusWidget"], [data-testid="stExpander"]');
        out.status_count = document.querySelectorAll('[data-testid="stStatusWidget"]').length;
        return out;
    }"""
    )


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        msgs = []
        page.on("console", lambda m: msgs.append(m.text))
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_selector('[data-testid="stFileUploader"]', timeout=30000)
        await asyncio.sleep(2)
        snap_before = await snapshot_struct(page)
        print("=== BEFORE UPLOAD ===")
        print(json.dumps(snap_before, indent=2, ensure_ascii=False))

        # Upload the file
        uploader = page.locator('[data-testid="stFileUploader"] input[type="file"]')
        await uploader.set_input_files(PDF)
        # Capture rapidly after upload (the flash frame) - poll a few times
        for i in range(6):
            await asyncio.sleep(0.25)
            snap = await snapshot_struct(page)
            print(f"=== AFTER UPLOAD t+{(i + 1) * 0.25}s ===")
            print(json.dumps(snap, indent=2, ensure_ascii=False))

        # Wait for analysis to complete
        await asyncio.sleep(6)
        snap_done = await snapshot_struct(page)
        print("=== AFTER ANALYSIS (~6s) ===")
        print(json.dumps(snap_done, indent=2, ensure_ascii=False))

        await browser.close()


asyncio.run(main())
