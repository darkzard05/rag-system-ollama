"""Deep DOM analysis: find container structure and correct CSS targets."""
import asyncio
import json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501", wait_until="networkidle")
        await asyncio.sleep(4)

        # 1. Detailed tree structure of stMainBlockContainer
        tree = await page.evaluate("""() => {
            const main = document.querySelector('[data-testid="stMainBlockContainer"]');
            if (!main) return { error: "NO MAIN" };
            
            function inspect(el, depth) {
                if (depth > 6 || !el) return null;
                const tid = el.getAttribute("data-testid") || "";
                const cs = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return {
                    tag: el.tagName,
                    testid: tid,
                    cls: (el.className || "").substring(0, 50),
                    height: cs.height,
                    maxHeight: cs.maxHeight,
                    overflowY: cs.overflowY,
                    display: cs.display,
                    position: cs.position,
                    rect: { t: Math.round(rect.top), b: Math.round(rect.bottom), h: Math.round(rect.height) },
                    nkids: el.children.length,
                    children: Array.from(el.children).slice(0, 6).map(c => inspect(c, depth + 1))
                };
            }
            return inspect(main, 0);
        }""")
        
        print("=== stMainBlockContainer TREE ===")
        print(json.dumps(tree, indent=2, default=str))
        
        # 2. Check for scrollable containers
        scrollable = await page.evaluate("""() => {
            const all = document.querySelectorAll("*");
            const results = [];
            all.forEach(el => {
                const cs = window.getComputedStyle(el);
                if ((cs.overflowY === "auto" || cs.overflowY === "scroll") &&
                    parseInt(cs.height) > 50) {
                    const tid = el.getAttribute("data-testid") || "(no testid)";
                    const rect = el.getBoundingClientRect();
                    results.push({
                        testid: tid,
                        tag: el.tagName,
                        height: cs.height,
                        overflowY: cs.overflowY,
                        scrollH: el.scrollHeight,
                        clientH: el.clientHeight,
                        rect: { t: Math.round(rect.top), b: Math.round(rect.bottom) }
                    });
                }
            });
            return results;
        }""")
        
        print("\n=== SCROLLABLE CONTAINERS (overflow-y: auto/scroll) ===")
        for s in scrollable:
            print(f"  {s['testid']}: tag={s['tag']} height={s['height']} overflow={s['overflowY']}")
            print(f"    rect top={s['rect']['t']} bottom={s['rect']['b']} scrollH={s['scrollH']}")
        
        # 3. Check for elements with inline height style (from st.container(height=X))
        inline_height = await page.evaluate("""() => {
            const all = document.querySelectorAll("[style]");
            const results = [];
            all.forEach(el => {
                const style = el.getAttribute("style") || "";
                if (style.includes("height")) {
                    const tid = el.getAttribute("data-testid") || "(no testid)";
                    results.push({
                        testid: tid,
                        tag: el.tagName,
                        style: style.substring(0, 100)
                    });
                }
            });
            return results;
        }""")
        
        print("\n=== ELEMENTS WITH INLINE HEIGHT ===")
        for r in inline_height:
            print(f"  {r['testid']}: tag={r['tag']} style='{r['style']}'")
        
        await browser.close()

asyncio.run(main())
