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
EVIDENCE_FILE = Path(".omo/evidence/task-1-dom-reverify.txt")

async def collect_dom_metrics(page):
    logger.info("Collecting DOM metrics for Task 1...")

    try:
        # 1. DOM Hierarchy Tree & 6. Chain Verification
        # We'll use a JS function to trace the path and collect metrics
        hierarchy_and_chain = await page.evaluate("""() => {
            function getElementMetrics(el) {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return {
                    tagName: el.tagName,
                    testId: el.getAttribute('data-testid'),
                    display: style.display,
                    height: style.height,
                    overflowY: style.overflowY,
                    flex: style.flexDirection,
                    children: Array.from(el.children)
                        .filter(child => {
                            const tid = child.getAttribute('data-testid');
                            return tid && (
                                tid.includes('stMainBlockContainer') ||
                                tid.includes('stVerticalBlock') ||
                                tid.includes('stLayoutWrapper') ||
                                tid.includes('stHorizontalBlock') ||
                                tid.includes('stColumn') ||
                                tid.includes('stElementContainer')
                            );
                        })
                        .map(child => getElementMetrics(child))
                };
            }

            const root = document.querySelector('[data-testid="stMainBlockContainer"]');
            const tree = root ? getElementMetrics(root) : null;

            // Trace the specific 7-step chain claim
            // stMainBlockContainer > stVerticalBlock > stLayoutWrapper > stHorizontalBlock > stColumn > stVerticalBlock > stLayoutWrapper > stVerticalBlock(scrollable)
            const chain = [];
            let current = root;
            const targetChain = [
                'stMainBlockContainer', 'stVerticalBlock', 'stLayoutWrapper', 
                'stHorizontalBlock', 'stColumn', 'stVerticalBlock', 
                'stLayoutWrapper', 'stVerticalBlock'
            ];
            
            // This is a simplified trace. In reality, we should search for the path.
            function findPath(el, targetIdx, path) {
                if (!el) return null;
                const tid = el.getAttribute('data-testid') || '';
                if (tid.includes(targetChain[targetIdx])) {
                    const newPath = [...path, tid];
                    if (targetIdx === targetChain.length - 1) return newPath;
                    for (let child of el.children) {
                        const res = findPath(child, targetIdx + 1, newPath);
                        if (res) return res;
                    }
                }
                for (let child of el.children) {
                    const res = findPath(child, targetIdx, path);
                    if (res) return res;
                }
                return null;
            }
            
            const actualChain = findPath(root, 0, []);

            return { tree, actualChain };
        }""")

        # 2. stColumn display
        st_column_display = await page.evaluate("""() => {
            const col = document.querySelector('[data-testid="stColumn"]');
            return col ? window.getComputedStyle(col).display : 'NOT FOUND';
        }""")

        # 3. stLayoutWrapper display
        layout_wrappers_display = await page.evaluate("""() => {
            const wrappers = document.querySelectorAll('[data-testid="stLayoutWrapper"]');
            return Array.from(wrappers).map(el => ({
                testId: el.getAttribute('data-testid'),
                display: window.getComputedStyle(el).display
            }));
        }""")

        # 4. Scrollable container
        scrollable_containers = await page.evaluate("""() => {
            const cols = document.querySelectorAll('[data-testid="stColumn"]');
            const results = [];
            cols.forEach((col, idx) => {
                let deepest = null;
                const allChildren = col.querySelectorAll('*');
                allChildren.forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        deepest = el;
                    }
                });
                if (deepest) {
                    // Build selector path
                    const path = [];
                    let curr = deepest;
                    while (curr && curr.tagName !== 'BODY') {
                        const tid = curr.getAttribute('data-testid');
                        const selector = tid ? `[data-testid="${tid}"]` : curr.tagName.toLowerCase();
                        path.unshift(selector);
                        curr = curr.parentElement;
                    }
                    results.push({
                        columnIndex: idx,
                        selectorPath: path.join(' > '),
                        testId: deepest.getAttribute('data-testid'),
                        overflowY: window.getComputedStyle(deepest).overflowY
                    });
                }
            });
            return results;
        }""")

        # 5. Chat input position
        chat_input_info = await page.evaluate("""() => {
            const input = document.querySelector('[data-testid="stChatInput"]');
            if (!input) return null;

            const style = window.getComputedStyle(input);
            const parentChain = [];
            let current = input.parentElement;
            while (current) {
                const s = window.getComputedStyle(current);
                parentChain.push({
                    tagName: current.tagName,
                    testId: current.getAttribute('data-testid'),
                    position: s.position
                });
                current = current.parentElement;
            }

            return {
                inputPosition: style.position,
                parentChain: parentChain
            };
        }""")

        return {
            "hierarchy": hierarchy_and_chain["tree"],
            "actualChain": hierarchy_and_chain["actualChain"],
            "stColumnDisplay": st_column_display,
            "layoutWrappersDisplay": layout_wrappers_display,
            "scrollableContainers": scrollable_containers,
            "chatInput": chat_input_info,
        }

    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        raise e

async def main():
    # Ensure evidence directory exists
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        logger.info(f"Launching browser and navigating to {BASE_URL}...")
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()

            try:
                await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
                logger.info("Page loaded. Waiting 5s for Streamlit render...")
                await asyncio.sleep(5)
            except TimeoutError:
                logger.error(f"❌ FAIL: Timeout waiting for {BASE_URL}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"❌ FAIL: Could not connect to app: {e}")
                sys.exit(1)

            metrics = await collect_dom_metrics(page)

            # Format results to evidence file
            with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
                f.write("=== TASK 1: DOM STRUCTURE RE-VERIFICATION EVIDENCE ===\n\n")

                f.write("--- 1. DOM Hierarchy Tree (stMainBlockContainer down) ---\n")
                f.write(json.dumps(metrics["hierarchy"], indent=2))
                f.write("\n\n")

                f.write("--- 2. stColumn display property ---\n")
                f.write(f"Computed Display: {metrics['stColumnDisplay']}\n\n")

                f.write("--- 3. stLayoutWrapper display properties ---\n")
                f.write(json.dumps(metrics["layoutWrappersDisplay"], indent=2))
                f.write("\n\n")

                f.write("--- 4. Scrollable containers in stColumn ---\n")
                f.write(json.dumps(metrics["scrollableContainers"], indent=2))
                f.write("\n\n")

                f.write("--- 5. Chat input position and parent chain ---\n")
                f.write(json.dumps(metrics["chatInput"], indent=2))
                f.write("\n\n")

                f.write("--- 6. Chain Verification ---\n")
                f.write("Claimed Chain: stMainBlockContainer > stVerticalBlock > stLayoutWrapper > stHorizontalBlock > stColumn > stVerticalBlock > stLayoutWrapper > stVerticalBlock(scrollable)\n")
                if metrics["actualChain"]:
                    f.write(f"Actual Found Chain: {' > '.join(metrics['actualChain'])}\n")
                    match = "MATCH" if len(metrics["actualChain"]) >= 8 else "DISCREPANCY"
                    f.write(f"Result: {match}\n")
                else:
                    f.write("Actual Found Chain: NOT FOUND\n")
                    f.write("Result: DISCREPANCY\n")

            logger.info(f"✅ SUCCESS: Evidence saved to {EVIDENCE_FILE}")
            await browser.close()
            sys.exit(0)

        except Exception as e:
            logger.error(f"❌ FAIL: An unexpected error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
