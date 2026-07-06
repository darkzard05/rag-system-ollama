import asyncio
import json
import logging
import sys
from pathlib import Path

from playwright.async_api import TimeoutError, async_playwright

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://127.0.0.1:8501"
TIMEOUT = 30000  # 30 seconds
EVIDENCE_FILE = Path(".omo/evidence/task-1-column-dom-structure.txt")


async def collect_dom_metrics(page):
    """
    Collects 6 specific DOM properties from the Streamlit application.
    """
    logger.info("Collecting DOM metrics...")

    try:
        # Item 1: DOM Hierarchy Tree
        # We use a recursive function in JS to capture the structure
        hierarchy_data = await page.evaluate("""() => {
            function getElementMetrics(el) {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return {
                    testId: el.getAttribute('data-testid'),
                    tagName: el.tagName,
                    display: style.display,
                    height: style.height,
                    maxHeight: style.maxHeight,
                    overflowY: style.overflowY,
                    rect: {
                        top: rect.top,
                        bottom: rect.bottom,
                        height: rect.height
                    },
                    childrenCount: el.children.length,
                    children: Array.from(el.children)
                        .filter(child => {
                            // Only track relevant Streamlit containers to avoid noise
                            const tid = child.getAttribute('data-testid');
                            return tid && (
                                tid.includes('stMainBlockContainer') ||
                                tid.includes('stVerticalBlock') ||
                                tid.includes('stElementContainer') ||
                                tid.includes('stLayoutWrapper') ||
                                tid.includes('stHorizontalBlock') ||
                                tid.includes('stColumn')
                            );
                        })
                        .map(child => getElementMetrics(child))
                };
            }

            const root = document.querySelector('[data-testid="stMainBlockContainer"]');
            return root ? getElementMetrics(root) : null;
        }""")

        # Item 2: st.container() DOM structure
        container_heights = await page.evaluate("""() => {
            const cols = document.querySelectorAll('[data-testid="stColumn"]');
            const results = [];
            cols.forEach(col => {
                // Look for elements with inline height styles inside columns
                const elements = col.querySelectorAll('*');
                elements.forEach(el => {
                    if (el.style.height && el.style.height !== 'auto') {
                        results.push({
                            testId: el.getAttribute('data-testid'),
                            inlineStyle: el.getAttribute('style'),
                            computedStyle: window.getComputedStyle(el).height
                        });
                    }
                });
            });
            return results;
        }""")

        # Item 3: Header actual height
        header_height = await page.evaluate("""() => {
            const header = document.querySelector('[data-testid="stHeader"]');
            if (!header) return null;
            const rect = header.getBoundingClientRect();
            return rect.height;
        }""")

        # Item 4: stLayoutWrapper display property
        layout_wrapper_display = await page.evaluate("""() => {
            const wrappers = document.querySelectorAll('[data-testid="stLayoutWrapper"]');
            return Array.from(wrappers).map(el => ({
                testId: el.getAttribute('data-testid'),
                display: window.getComputedStyle(el).display
            }));
        }""")

        # Item 5: st.chat_input rendering location
        chat_input_info = await page.evaluate("""() => {
            const input = document.querySelector('[data-testid="stChatInput"]');
            if (!input) return null;

            const style = window.getComputedStyle(input);
            const parentChain = [];
            let current = input.parentElement;
            while (current) {
                parentChain.push({
                    tagName: current.tagName,
                    testId: current.getAttribute('data-testid')
                });
                current = current.parentElement;
            }

            return {
                position: style.position,
                bottom: style.bottom,
                parentChain: parentChain
            };
        }""")

        # Item 6: Column inner scrollable container
        inner_scrollable = await page.evaluate("""() => {
            const cols = document.querySelectorAll('[data-testid="stColumn"]');
            const results = [];
            cols.forEach((col, idx) => {
                // Find the deepest element that has overflow-y: auto or could accept it
                let deepest = null;
                const allChildren = col.querySelectorAll('*');
                allChildren.forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        deepest = el;
                    }
                });

                if (deepest) {
                    const style = window.getComputedStyle(deepest);
                    results.push({
                        columnIndex: idx,
                        testId: deepest.getAttribute('data-testid'),
                        tagName: deepest.tagName,
                        overflowY: style.overflowY,
                        height: style.height,
                        maxHeight: style.maxHeight
                    });
                }
            });
            return results;
        }""")

        return {
            "item1_hierarchy": hierarchy_data,
            "item2_containers": container_heights,
            "item3_header_height": header_height,
            "item4_layout_display": layout_wrapper_display,
            "item5_chat_input": chat_input_info,
            "item6_inner_scrollable": inner_scrollable,
        }

    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        raise e


async def main():
    async with async_playwright() as p:
        logger.info(f"Launching browser and navigating to {BASE_URL}...")
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
                logger.info("Page loaded. Waiting 3s for Streamlit render...")
                await asyncio.sleep(3)
            except TimeoutError:
                logger.error(f"❌ FAIL: Timeout waiting for {BASE_URL}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"❌ FAIL: Could not connect to app: {e}")
                sys.exit(1)

            metrics = await collect_dom_metrics(page)

            # Format results to evidence file
            with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
                f.write("=== DOM STRUCTURE VERIFICATION EVIDENCE ===\n\n")

                f.write("--- Item 1: DOM Hierarchy Tree ---\n")
                f.write(json.dumps(metrics["item1_hierarchy"], indent=2))
                f.write("\n\n")

                f.write("--- Item 2: st.container() DOM structure ---\n")
                f.write(json.dumps(metrics["item2_containers"], indent=2))
                f.write("\n\n")

                f.write("--- Item 3: Header actual height ---\n")
                f.write(f"Height: {metrics['item3_header_height']}px\n\n")

                f.write("--- Item 4: stLayoutWrapper display property ---\n")
                f.write(json.dumps(metrics["item4_layout_display"], indent=2))
                f.write("\n\n")

                f.write("--- Item 5: st.chat_input rendering location ---\n")
                f.write(json.dumps(metrics["item5_chat_input"], indent=2))
                f.write("\n\n")

                f.write("--- Item 6: Column inner scrollable container ---\n")
                f.write(json.dumps(metrics["item6_inner_scrollable"], indent=2))
                f.write("\n\n")

            logger.info(f"✅ SUCCESS: Evidence saved to {EVIDENCE_FILE}")
            await browser.close()
            sys.exit(0)

        except Exception as e:
            logger.error(f"❌ FAIL: An unexpected error occurred: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
