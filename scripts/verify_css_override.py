"""Verify CSS height override is working on column containers."""
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
        
        print("2. Uploading test PDF...")
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(PDF_PATH)
        await asyncio.sleep(8)
        
        # Deep analysis: trace the height source for each stVerticalBlock in the columns
        print("\n3. Deep CSS analysis of column containers...")
        result = await page.evaluate("""() => {
            const mainBlock = document.querySelector('[data-testid="stMainBlockContainer"]');
            if (!mainBlock) return { error: 'no mainBlock' };
            
            const columns = mainBlock.querySelectorAll(':scope [data-testid="stColumn"]');
            const results = [];
            
            columns.forEach((col, colIdx) => {
                // Find ALL stVerticalBlocks inside this column at any depth
                const allVBs = col.querySelectorAll('[data-testid="stVerticalBlock"]');
                allVBs.forEach((vb, vbIdx) => {
                    const cs = window.getComputedStyle(vb);
                    const parent = vb.parentElement;
                    const parentTid = parent ? parent.getAttribute('data-testid') : 'none';
                    const styleAttr = vb.getAttribute('style') || '(no inline style)';
                    
                    // Check if our selector matches
                    const layoutWrapper = vb.closest('[data-testid="stLayoutWrapper"]');
                    const isDirectChild = layoutWrapper && layoutWrapper.children[0] === vb;
                    
                    results.push({
                        colIdx: colIdx,
                        vbIdx: vbIdx,
                        testid: vb.getAttribute('data-testid'),
                        parentTid: parentTid,
                        height: cs.height,
                        maxHeight: cs.maxHeight,
                        overflowY: cs.overflowY,
                        isDirectChildOfLayoutWrapper: isDirectChild,
                        styleAttr: styleAttr.substring(0, 150),
                        rect: vb.getBoundingClientRect()
                    });
                });
            });
            
            return results;
        }""")
        
        print(f"\nFound {len(result)} stVerticalBlock elements across columns:")
        for r in result:
            marker = "✅" if r['isDirectChildOfLayoutWrapper'] else "  "
            print(f"  {marker} Col{r['colIdx']}/VB{r['vbIdx']}: "
                  f"parent={r['parentTid']} height={r['height']} "
                  f"overflow={r['overflowY']} directChild={r['isDirectChildOfLayoutWrapper']}")
            if r['styleAttr'] != '(no inline style)':
                print(f"       style='{r['styleAttr']}'")
            if r['isDirectChildOfLayoutWrapper']:
                print(f"       rect top={r['rect']['top']} bottom={r['rect']['bottom']}")
        
        # Try alternative selectors to find what actually has viewport-relative height
        print("\n4. Trying alternative selectors...")
        selectors = [
            ("stCol > stLayout > stVB", 
             '[data-testid="stColumn"] [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]'),
            ("stCol > stVB", 
             '[data-testid="stColumn"] > [data-testid="stVerticalBlock"]'),
            ("stCol > stEC > stVB", 
             '[data-testid="stColumn"] > [data-testid="stElementContainer"] [data-testid="stVerticalBlock"]'),
            ("stCol stEC stVB", 
             '[data-testid="stColumn"] [data-testid="stElementContainer"] [data-testid="stVerticalBlock"]'),
        ]
        
        for label, selector in selectors:
            count = await page.evaluate(f"""() => {{
                const els = document.querySelectorAll('{selector}');
                return els.length;
            }}""")
            print(f"  {label}: {selector}")
            print(f"    -> matches {count} element(s)")
            
            if count > 0:
                heights = await page.evaluate(f"""() => {{
                    const els = document.querySelectorAll('{selector}');
                    return Array.from(els).map(el => ({{
                        height: window.getComputedStyle(el).height,
                        maxHeight: window.getComputedStyle(el).maxHeight,
                        overflowY: window.getComputedStyle(el).overflowY
                    }}));
                }}""")
                for h in heights:
                    print(f"    height={h['height']} maxHeight={h['maxHeight']} overflow={h['overflowY']}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
