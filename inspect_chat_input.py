import asyncio

from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8501")

        # Wait for the chat input to appear
        try:
            await page.wait_for_selector('[data-testid="stChatInput"]', timeout=10000)
        except Exception as e:
            print(f"Error waiting for selector: {e}")
            await browser.close()
            return

        # 1. Inspect the target element
        element = page.locator('[data-testid="stChatInput"]')

        # Get computed styles
        styles = await element.evaluate("""
            el => {
                const computed = window.getComputedStyle(el);
                return {
                    position: computed.position,
                    bottom: computed.bottom,
                    left: computed.left,
                    right: computed.right,
                    width: computed.width,
                    zIndex: computed.zIndex,
                    display: computed.display
                };
            }
        """)
        print(f"Target Element Computed Styles: {styles}")

        # 2. Inspect parents to find containing blocks
        parents_info = await element.evaluate("""
            el => {
                const parents = [];
                let current = el.parentElement;
                while (current) {
                    const style = window.getComputedStyle(current);
                    parents.push({
                        tagName: current.tagName,
                        id: current.id,
                        className: current.className,
                        testId: current.getAttribute('data-testid'),
                        position: style.position,
                        transform: style.transform,
                        filter: style.filter,
                        perspective: style.perspective
                    });
                    current = current.parentElement;
                }
                return parents;
            }
        """)

        print("\nParent Elements Analysis:")
        for i, p in enumerate(parents_info):
            print(
                f"Level {i + 1}: {p['tagName']} | testId: {p['testId']} | pos: {p['position']} | transform: {p['transform']}"
            )

        # 3. Dump the DOM structure around the chat input
        dom_dump = await element.evaluate("""
            el => {
                return el.closest('div[data-testid="stAppViewContainer"]').outerHTML;
            }
        """)
        with open("dom_dump.html", "w", encoding="utf-8") as f:
            f.write(dom_dump)
        print("\nDOM dump saved to dom_dump.html")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
