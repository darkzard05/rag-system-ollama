"""Final verification: Upload PDF, render main layout, verify DOM structure."""
import asyncio
import sys
from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8501"
PDF_PATH = r"C:\Users\darkzard05\hy\rag-system-ollama\tests\data\2201.07520v1.pdf"

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        print("1. Navigating to app...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)
        
        # 2. Upload PDF
        print("2. Uploading test PDF...")
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(PDF_PATH)
        await asyncio.sleep(8)  # Wait for PDF processing
        
        # 3. Dump main layout data-testid
        print("3. Checking main layout DOM...")
        testids = await page.evaluate("""() => {
            const all = document.querySelectorAll("[data-testid]");
            const counts = {};
            all.forEach(el => {
                const tid = el.getAttribute("data-testid");
                counts[tid] = (counts[tid] || 0) + 1;
            });
            return Object.entries(counts).sort((a,b)=>a[0].localeCompare(b[0]));
        }""")
        
        print("\n=== DATA-TESTID COUNTS ===")
        key_items = ["stMainBlockContainer", "stColumn", "stHorizontalBlock", 
                     "stLayoutWrapper", "stVerticalBlock", "stChatInput",
                     "stChatMessage", "stElementContainer"]
        found_keys = {k: v for k, v in testids if k in key_items}
        for k in key_items:
            if k in found_keys:
                print(f"  ✅ {k}: {found_keys[k]}")
            else:
                print(f"  ❌ {k}: NOT FOUND")
        
        # 4. Check stLayoutWrapper > stVerticalBlock structure
        print("\n4. Checking stLayoutWrapper > stVerticalBlock structure...")
        layout_blocks = await page.evaluate("""() => {
            const wrappers = document.querySelectorAll(
                '[data-testid=\"stMainBlockContainer\"] [data-testid=\"stColumn\"] ' +
                '[data-testid=\"stLayoutWrapper\"]'
            );
            return Array.from(wrappers).map((w, i) => {
                const vb = w.querySelector(':scope > [data-testid=\"stVerticalBlock\"]');
                if (!vb) return { idx: i, hasVerticalBlock: false };
                const cs = window.getComputedStyle(vb);
                return {
                    idx: i,
                    hasVerticalBlock: true,
                    overflowY: cs.overflowY,
                    height: cs.height,
                    maxHeight: cs.maxHeight,
                    display: cs.display,
                    parentTestId: w.parentElement?.getAttribute('data-testid') || 'unknown'
                };
            });
        }""")
        
        for lb in layout_blocks:
            if lb.get("hasVerticalBlock"):
                print(f"  Column {lb['idx']}: overflowY={lb['overflowY']} height={lb['height']} "
                      f"maxHeight={lb['maxHeight']} parent={lb['parentTestId']}")
            else:
                print(f"  Column {lb['idx']}: NO stVerticalBlock inside stLayoutWrapper")
        
        # 5. Check right column padding
        print("\n5. Checking right column padding...")
        right_padding = await page.evaluate("""() => {
            const columns = document.querySelectorAll(
                '[data-testid=\"stMainBlockContainer\"] [data-testid=\"stColumn\"]:last-child ' +
                '[data-testid=\"stLayoutWrapper\"] > [data-testid=\"stVerticalBlock\"]'
            );
            if (columns.length === 0) return false;
            const last = columns[columns.length - 1];
            const cs = window.getComputedStyle(last);
            return { paddingBottom: cs.paddingBottom };
        }""")
        if right_padding:
            print(f"  Right column padding-bottom: {right_padding['paddingBottom']}")
        else:
            print("  No right column stVerticalBlock found")
        
        # 6. Body overflow check
        print("\n6. Checking body/app overflow...")
        overflow = await page.evaluate("""() => {
            const body = window.getComputedStyle(document.body);
            const app = document.querySelector('.stApp');
            const appStyle = app ? window.getComputedStyle(app) : null;
            return {
                bodyOverflow: body.overflow,
                appOverflow: appStyle ? appStyle.overflow : 'N/A',
                scrollY: window.scrollY
            };
        }""")
        print(f"  Body overflow: {overflow['bodyOverflow']}")
        print(f"  App overflow: {overflow['appOverflow']}")
        print(f"  scrollY: {overflow['scrollY']}")
        
        # 7. Chat input position
        print("\n7. Checking chat input position...")
        chat_pos = await page.evaluate("""() => {
            const chat = document.querySelector('[data-testid=\"stChatInput\"]');
            if (!chat) return null;
            const cs = window.getComputedStyle(chat);
            return {
                position: cs.position,
                left: cs.left,
                width: cs.width,
                bottom: cs.bottom
            };
        }""")
        if chat_pos:
            print(f"  Position: {chat_pos['position']}")
            print(f"  Left: {chat_pos['left']}")
            print(f"  Width: {chat_pos['width']}")
        else:
            print("  stChatInput NOT FOUND")
        
        # Overall verdict
        print("\n" + "=" * 50)
        c1 = found_keys.get("stMainBlockContainer", 0) >= 1
        c2 = found_keys.get("stColumn", 0) >= 2
        c3 = found_keys.get("stChatInput", 0) >= 1
        c4 = any(lb.get("hasVerticalBlock") and lb.get("overflowY") in ("auto", "scroll") for lb in layout_blocks)
        c5 = overflow.get("scrollY", 1) == 0
        
        print(f"  C1: stMainBlockContainer exists: {'✅' if c1 else '❌'}")
        print(f"  C2: stColumn exists (>=2): {'✅' if c2 else '❌'}")
        print(f"  C3: stChatInput exists: {'✅' if c3 else '❌'}")
        print(f"  C4: Scrollable containers: {'✅' if c4 else '❌'}")
        print(f"  C5: Body locked (scrollY=0): {'✅' if c5 else '❌'}")
        
        all_pass = c1 and c2 and c3 and c4 and c5
        print(f"\n  VERDICT: {'✅ ALL PASSED' if all_pass else '❌ SOME FAILED'}")
        print("=" * 50)
        
        await browser.close()
        sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    asyncio.run(verify())
