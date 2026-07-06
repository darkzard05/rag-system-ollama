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

            content = await page.content()
            
            print("\n--- Search Results ---")
            
            found_pdf = False
            if 'class="pdf-viewer-container"' in content or 'class=\'pdf-viewer-container\'' in content:
                print("FOUND PDF: pdf-viewer-container")
                found_pdf = True
            
            found_chat = False
            if 'class="chat-container"' in content or 'class=\'chat-container\'' in content:
                print("FOUND CHAT: chat-container")
                found_chat = True

            if not found_pdf:
                print("PDF container NOT found in HTML.")
            if not found_chat:
                print("Chat container NOT found in HTML.")
            print("----------------------\n")

            # Also search for any element that has these classes in its class attribute
            # using a more robust way.
            
            elements_info = await page.evaluate("""() => {
                const results = [];
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    if (el.className && typeof el.className === 'string' && el.className.trim() !== '') {
                        results.push({
                            tagName: el.tagName,
                            className: el.className
                        });
                    }
                }
                return results;
            }""")

            print(f"Total elements with classes: {len(elements_info)}")
            print("-" * 30)

            for info in elements_info:
                if "pdf-viewer-container" in info['className']:
                    print(f"FOUND PDF: Tag: {info['tagName']}, Class: {info['className']}")
                if "chat-container" in info['className']:
                    print(f"FOUND CHAT: Tag: {info['tagName']}, Class: {info['className']}")
            print("-" * 30)

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
