import asyncio
import sys
from playwright.async_api import async_playwright


async def test_scroll_behavior():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        print("Navigating to Streamlit app...")
        try:
            await page.goto("http://localhost:8501", timeout=15000)
        except Exception as e:
            print(f"Error navigating to app: {e}")
            print("Make sure 'streamlit run src/main.py' is running in the background.")
            await browser.close()
            sys.exit(1)

        print("Waiting for columns to render...")
        try:
            await page.wait_for_selector(
                'div[data-testid="stColumn"], div[data-testid="column"]',
                state="attached",
                timeout=15000,
            )
        except Exception as e:
            print(f"Timeout waiting for columns: {e}")
            await browser.close()
            sys.exit(1)

        # Debug DOM Path
        path_script = """
            () => {
                const col = document.querySelector('div[data-testid="stColumn"]');
                let path = [];
                let current = col;
                while (current && current.tagName !== 'BODY') {
                    path.push(`${current.tagName}.${current.className} [${current.getAttribute('data-testid') || 'no-testid'}]`);
                    current = current.parentElement;
                }
                return path.join(' -> ');
            }
        """
        dom_path = await page.evaluate(path_script)
        print(f"DOM Path: {dom_path}")

        # 1. Global Viewport Lock Verification

        print("Verifying Global Viewport Lock...")
        body_overflow = await page.evaluate(
            "window.getComputedStyle(document.body).overflow"
        )
        st_app_overflow = await page.evaluate(
            "window.getComputedStyle(document.querySelector('.stApp')).overflow"
        )

        assert body_overflow == "hidden", (
            f"Body overflow should be hidden, got {body_overflow}"
        )
        assert st_app_overflow == "hidden", (
            f".stApp overflow should be hidden, got {st_app_overflow}"
        )
        print("✅ Global Viewport Lock verified.")

        # 2. Column Scroll Configuration Verification
        print("Verifying Column Scroll Configuration...")
        # target only columns inside the main block-container to avoid sidebar columns
        columns = await page.query_selector_all(
            '.block-container div[data-testid="stColumn"], .block-container div[data-testid="column"]'
        )
        if len(columns) < 2:
            print(
                f"Error: Expected at least 2 columns in main container, found {len(columns)}"
            )
            await browser.close()
            sys.exit(1)

        for i, col in enumerate(columns):
            overflow_y = await col.evaluate(
                "el => window.getComputedStyle(el).overflowY"
            )
            assert overflow_y in ["auto", "scroll"], (
                f"Column {i} should have overflow-y: auto/scroll, got {overflow_y}"
            )
        print("✅ Column Scroll Configuration verified.")

        # 3. Functional Independent Scroll Verification
        print("Verifying Independent Scroll Functionality...")

        # Perform injection, scroll, and measurement in ONE atomic JS call to prevent Streamlit re-renders
        results = await page.evaluate("""
            () => {
                const cols = document.querySelectorAll('.block-container div[data-testid="stColumn"], .block-container div[data-testid="column"]');
                if (cols.length < 2) return { error: 'Less than 2 columns found' };

                // Inject dummy content
                cols.forEach(col => {
                    const verticalBlock = col.querySelector('div[data-testid="stVerticalBlock"]');
                    const target = verticalBlock || col;
                    const dummy = document.createElement('div');
                    dummy.style.height = '5000px';
                    dummy.style.width = '100%';
                    dummy.innerText = 'Dummy content to force scroll';
                    target.appendChild(dummy);
                });

                // Scroll the first column
                cols[0].scrollTop = 500;

                return {
                    leftScrollTop: cols[0].scrollTop,
                    rightScrollTop: cols[1].scrollTop,
                    leftScrollHeight: cols[0].scrollHeight,
                    leftClientHeight: cols[0].clientHeight,
                    rightScrollHeight: cols[1].scrollHeight,
                    rightClientHeight: cols[1].clientHeight
                };
            }
        """)

        if "error" in results:
            print(f"Error: {results['error']}")
            await browser.close()
            sys.exit(1)

        print(
            f"Left: SH={results['leftScrollHeight']}, CH={results['leftClientHeight']}, ST={results['leftScrollTop']}"
        )
        print(
            f"Right: SH={results['rightScrollHeight']}, CH={results['rightClientHeight']}, ST={results['rightScrollTop']}"
        )

        assert results["leftScrollTop"] > 0, (
            f"Left column should have been scrolled, but scrollTop is {results['leftScrollTop']}"
        )
        assert results["rightScrollTop"] == 0, (
            f"Right column should NOT have scrolled, but scrollTop is {results['rightScrollTop']}"
        )
        print("✅ Independent Scroll Functionality verified.")

        # Scroll the first column and check the second
        await columns[0].evaluate("el => el.scrollTop = 500")

        # Wait a bit for rendering
        await asyncio.sleep(0.5)

        left_scroll = await columns[0].evaluate("el => el.scrollTop")
        right_scroll = await columns[1].evaluate("el => el.scrollTop")

        print(
            f"Left Column scrollTop: {left_scroll}, Right Column scrollTop: {right_scroll}"
        )

        assert left_scroll > 0, (
            f"Left column should have been scrolled, but scrollTop is {left_scroll}"
        )
        assert right_scroll == 0, (
            f"Right column should NOT have scrolled, but scrollTop is {right_scroll}"
        )
        print("✅ Independent Scroll Functionality verified.")

        await browser.close()
        print("\nALL SCROLL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(test_scroll_behavior())
