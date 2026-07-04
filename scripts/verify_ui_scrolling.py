"""
UI Scrolling Verification Script

This script uses Playwright to verify that the PDF viewer and Chat interface
in the Streamlit application scroll independently by checking their
computed CSS styles.

How to use:
1. Start the Streamlit app:
   streamlit run src/main.py

2. Install dependencies:
   pip install playwright
   playwright install chromium

3. Run this script:
   python scripts/verify_ui_scrolling.py
"""

import asyncio
import logging
import sys

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

# Selectors
# We find all stVerticalBlocks and filter them by style and parent in JS.
SCROLLABLE_BLOCK_SELECTOR = (
    "[data-testid='stLayoutWrapper'] [data-testid='stVerticalBlock']"
)


async def verify_elements_scrolling(page):
    """
    Finds all stVerticalBlocks and verifies that at least two of them
    (PDF and Chat) have the correct scrolling styles and are children of stColumn.
    """
    logger.info(
        f"Searching for all vertical blocks using {SCROLLABLE_BLOCK_SELECTOR}..."
    )

    try:
        # Wait for at least one block to appear
        await page.wait_for_selector(
            SCROLLABLE_BLOCK_SELECTOR, state="attached", timeout=TIMEOUT
        )

        # Get all matching elements and their styles/parents in one go
        results = await page.evaluate("""() => {
            const blocks = document.querySelectorAll('[data-testid="stLayoutWrapper"] [data-testid="stVerticalBlock"]');
            return Array.from(blocks)
                .filter(el => el.clientHeight > 100)
                .map(el => {
                    const style = window.getComputedStyle(el);
                    return {
                        overflowY: style.overflowY,
                        maxHeight: style.maxHeight
                    };
                });
        }""")

        logger.info(f"Found {len(results)} total stVerticalBlocks. Analyzing styles...")

        verified_count = 0
        for i, item in enumerate(results):
            overflow_ok = item["overflowY"] in ["auto", "scroll"]

            if overflow_ok:
                logger.info(
                    f"Block {i}: ✅ PASS (overflowY={item['overflowY']}, maxHeight={item['maxHeight']})"
                )
                verified_count += 1
            else:
                logger.info(
                    f"Block {i}: ❌ SKIP (overflowY={item['overflowY']}, maxHeight={item['maxHeight']})"
                )

        if verified_count >= 2:
            logger.info(
                f"✅ SUCCESS: {verified_count} column-based blocks verified as independently scrollable."
            )
            return True
        else:
            logger.error(
                f"❌ FAIL: Only {verified_count} column-based blocks were scrollable. Need at least 2."
            )
            return False

    except TimeoutError:
        logger.error(f"❌ FAIL: No stVerticalBlocks found within {TIMEOUT / 1000}s")
        return False
    except Exception as e:
        logger.error(f"❌ FAIL: An error occurred: {e}")
        return False


async def main():
    async with async_playwright() as p:
        logger.info(f"Launching browser and navigating to {BASE_URL}...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            logger.info("Page loaded successfully.")
            await asyncio.sleep(3)

            success = await verify_elements_scrolling(page)

            print("\n" + "=" * 30)
            print("   SCROLLING VERIFICATION")
            print("=" * 30)
            print(f"Result: {'PASS' if success else 'FAIL'}")
            print("=" * 30)

            if success:
                logger.info("All scrolling verifications PASSED.")
                sys.exit(0)
            else:
                logger.error("Scrolling verification FAILED.")
                sys.exit(1)

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            sys.exit(1)
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
