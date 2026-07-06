import asyncio
import logging
import sys
from playwright.async_api import async_playwright, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8501"
TIMEOUT = 60000  # 60 seconds

async def get_css_selector(element):
    """
    Generates a CSS selector for the given element.
    """
    return await element.evaluate("""(el) => {
        if (el.id) return '#' + el.id;
        let path = [];
        while (el.nodeType === Node.ELEMENT_NODE) {
            let selector = el.tagName.toLowerCase();
            if (el.className && typeof el.className === 'string') {
                let classes = el.className.trim().split(/\s+/).join('.');
                if (classes) {
                    selector += '.' + classes;
                }
            }
            path.unshift(selector);
            el = el.parentNode;
            if (!el || el.nodeType !== Node.ELEMENT_NODE) break;
        }
        return path.join(' > ');
    }""")

async def main():
    async with async_playwright() as p:
        logger.info(f"Launching browser and navigating to {BASE_URL}...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            logger.info("Page loaded successfully.")
            
            # Give Streamlit a moment to stabilize the layout
            await asyncio.sleep(5)
            
            # 1. HTML Dump
            content = await page.content()
            with open("dom_dump.html", "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("HTML dumped to dom_dump.html")

            # 2. Computed Style Search
            logger.info("Searching for scrollable elements...")
            scrollable_elements_data = await page.evaluate("""() => {
                const results = [];
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    const style = window.getComputedStyle(el);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        results.push({
                            tagName: el.tagName,
                            className: el.className,
                            // We can't easily get the full selector here without a helper, 
                            // so we'll just return enough info to identify it.
                        });
                    }
                }
                return results;
            }""")

            # Since we need the full selector, we'll do it in a more controlled way
            # by iterating through the elements found in the previous step.
            
            # Re-run the search but this time we'll use Playwright to get the selector
            # to avoid complexity in the JS evaluate.
            
            # First, find all elements that match the criteria using a JS expression
            # and return their unique identifiers or just use the evaluate to find them.
            
            # Let's try a different approach: find all elements with overflowY auto/scroll
            # and then for each, get its details.
            
            # We'll use a JS function to find them and return their index in the document
            # or something similar. Actually, let's just use the evaluate to return 
            # a list of objects that we can then use to find the elements in Playwright.
            
            # Let's use a more robust way to find them.
            scrollable_info = await page.evaluate("""() => {
                const results = [];
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    const style = window.getComputedStyle(el);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        // We'll use a custom attribute to mark them for Playwright
                        const marker = 'data-scrollable-marker-' + Math.random().toString(36).substring(2, 9);
                        el.setAttribute('data-scrollable-marker', marker);
                        results.push({
                            marker: marker,
                            tagName: el.tagName,
                            className: typeof el.className === 'string' ? el.className : ''
                        });
                    }
                }
                return results;
            }""")

            logger.info(f"Found {len(scrollable_info)} scrollable elements.")

            for info in scrollable_info:
                marker = info['marker']
                element = page.locator(f'[data-scrollable-marker="{marker}"]')
                
                selector = await get_css_selector(await element.element_handle())
                
                print(f"Tag: {info['tagName']}, Class: {info['className']}, Selector: {selector}")

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
