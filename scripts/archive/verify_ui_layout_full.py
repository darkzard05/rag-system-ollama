import asyncio
import subprocess
import time
import os
from playwright.async_api import async_playwright


async def run_test():
    log_file = "streamlit_app.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            ["streamlit", "run", "src/main.py"], stdout=f, stderr=f, shell=True
        )

    print("Starting Streamlit app... Logging to streamlit_app.log")
    time.sleep(30)  # Give it more time

    # Find the port from the log file
    url = "http://localhost:8501"
    try:
        with open("streamlit_app.log", "r") as f:
            for line in f:
                if "Local URL:" in line:
                    url = line.split("Local URL:")[1].strip()
                    break
        print(f"Detected app URL: {url}")
    except Exception as e:
        print(f"Could not detect port from log, using default 8501. Error: {e}")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Test multiple resolutions
            resolutions = [
                {"width": 1920, "height": 1080},
                {"width": 1366, "height": 768},
                {"width": 1024, "height": 768},
            ]

            for res in resolutions:
                print(f"\nTesting resolution: {res['width']}x{res['height']}")
                context = await browser.new_context(viewport=res)
                page = await context.new_page()

                print(f"Connecting to app at {url}...")
                try:
                    await page.goto(url, timeout=60000)
                except Exception as e:
                    print(f"Error: Could not connect to app. {e}")
                    continue

                print("Waiting for app to load...")
                try:
                    await page.wait_for_function(
                        "() => document.querySelector('.stApp')?.getAttribute('data-test-script-state') !== 'running'",
                        timeout=60000,
                    )
                    await page.wait_for_selector(
                        '[data-testid="stAppViewContainer"]', timeout=30000
                    )

                    # Try multiple selectors for the chat input
                    chat_input_selectors = [
                        '[data-testid="stChatInputContainer"]',
                        'textarea[data-testid="stChatInput"]',
                        'div[data-testid="stChatInput"]',
                    ]

                    input_found = False
                    input_container = None
                    for selector in chat_input_selectors:
                        try:
                            input_container = await page.wait_for_selector(
                                selector, timeout=10000
                            )
                            print(f"Found chat input with selector: {selector}")
                            input_found = True
                            break
                        except:
                            continue

                    if not input_found:
                        print(
                            f"FAIL: Chat input not found at {res['width']}x{res['height']}"
                        )
                        await page.screenshot(
                            path=f"error_{res['width']}x{res['height']}.png"
                        )
                        continue

                    # Check visibility (Coordinates)
                    box = await input_container.bounding_box()
                    viewport = page.viewport_size
                    if box and viewport:
                        is_visible = (
                            box["x"] >= 0
                            and box["y"] >= 0
                            and (box["x"] + box["width"]) <= viewport["width"]
                            and (box["y"] + box["height"]) <= viewport["height"]
                        )

                        # Check if it's actually at the bottom
                        is_at_bottom = box["y"] > (viewport["height"] * 0.7)

                        print(
                            f"Chat input in viewport: {is_visible} (x: {box['x']}, y: {box['y']}, w: {box['width']}, h: {box['height']})"
                        )
                        print(f"Chat input is at bottom: {is_at_bottom}")

                        if not is_visible or not is_at_bottom:
                            await page.screenshot(
                                path=f"clipped_or_top_{res['width']}x{res['height']}.png"
                            )
                    else:
                        print("FAIL: Bounding box or viewport not found")

                    # Verify Global Scroll
                    body_overflow = await page.evaluate(
                        "window.getComputedStyle(document.body).overflow"
                    )
                    html_overflow = await page.evaluate(
                        "window.getComputedStyle(document.documentElement).overflow"
                    )
                    app_overflow = await page.evaluate(
                        "window.getComputedStyle(document.querySelector('.stApp')).overflow"
                    )
                    global_scroll_disabled = (
                        body_overflow == "hidden"
                        or html_overflow == "hidden"
                        or app_overflow == "hidden"
                    )
                    print(f"Global scroll disabled: {global_scroll_disabled}")

                except Exception as e:
                    print(
                        f"Error during verification at {res['width']}x{res['height']}: {e}"
                    )
                    await page.screenshot(
                        path=f"exception_{res['width']}x{res['height']}.png"
                    )

                await context.close()

            await browser.close()
    finally:
        process.terminate()


if __name__ == "__main__":
    asyncio.run(run_test())
