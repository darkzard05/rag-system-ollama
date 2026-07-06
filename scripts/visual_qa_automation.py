import asyncio
import os
import subprocess
import time
import shutil
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
STREAMLIT_CMD = ["streamlit", "run", "src/main.py"]
BASE_URL = "http://localhost:8501"
EVIDENCE_DIR = Path("qa_evidence")
DUMMY_PDF = "test_dummy.pdf"

async def run_qa():
    # 1. Setup
    if EVIDENCE_DIR.exists():
        shutil.rmtree(EVIDENCE_DIR)
    EVIDENCE_DIR.mkdir(parents=True)
    
    print("🚀 Starting Streamlit...")
    process = subprocess.Popen(STREAMLIT_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for Streamlit to be ready
    max_retries = 20
    ready = False
    for i in range(max_retries):
        try:
            # Try to connect to the URL
            import urllib.request
            with urllib.request.urlopen(BASE_URL, timeout=1) as response:
                if response.getcode() == 200:
                    ready = True
                    break
        except Exception:
            pass
        print(f"  Waiting for Streamlit... ({i+1}/{max_retries})")
        await asyncio.sleep(2)
    
    if not ready:
        print("❌ Streamlit failed to start.")
        process.terminate()
        return

    print("✅ Streamlit is ready.")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(BASE_URL)
            await page.wait_for_load_state("networkidle")

            # --- TEST 1: Fluid Chat Transition ---
            print("🧪 Testing Fluid Chat Transition...")
            chat_input_selector = "text=추가 질문을 입력하세요..." # Fallback to placeholder
            # Try to find the chat input more robustly
            try:
                chat_input = page.get_by_placeholder("추가 질문을 입력하세요...")
                if await chat_input.count() == 0:
                    chat_input = page.get_by_placeholder("...")
            except:
                chat_input = page.locator("[data-testid='stChatInput']")

            await chat_input.fill("Hello, tell me about yourself.")
            await chat_input.press("Enter")
            
            # Monitor streaming
            streaming_screenshots = []
            start_time = time.time()
            while time.time() - start_time < 30: # Max 30s for streaming
                # Check if assistant is generating (look for the cursor or pulse)
                # We look for the presence of the cursor '▌' in the last message
                content = await page.content()
                if "▌" in content:
                    timestamp = int(time.time() * 1000)
                    screenshot_path = EVIDENCE_DIR / f"chat_streaming_{timestamp}.png"
                    await page.screenshot(path=str(screenshot_path))
                    streaming_screenshots.append(screenshot_path)
                    await asyncio.sleep(0.5)
                else:
                    # If no cursor, check if it's still generating via status or other indicators
                    # For now, if no cursor, assume streaming finished or it's a slow start
                    if len(streaming_screenshots) > 0:
                        break
                    await asyncio.sleep(1)
            
            print(f"  Captured {len(streaming_screenshots)} streaming screenshots.")

            # --- TEST 2: Layout Integrity ---
            print("🧪 Testing Layout Integrity...")
            viewports = [
                {"name": "desktop", "width": 1920, "height": 1080},
                {"name": "tablet", "width": 1024, "height": 768},
                {"name": "mobile", "width": 375, "height": 667},
            ]
            
            for vp in viewports:
                print(f"  Testing viewport: {vp['name']} ({vp['width']}x{vp['height']})")
                await context.set_viewport_size({"width": vp["width"], "height": vp["height"]})
                await page.goto(BASE_URL)
                await page.wait_for_load_state("networkidle")
                await page.screenshot(path=str(EVIDENCE_DIR / f"layout_{vp['name']}.png"))

            # --- TEST 3: State Update Responsiveness ---
            print("🧪 Testing State Update Responsiveness...")
            # Upload dummy PDF
            print("  Uploading dummy PDF...")
            file_input = page.locator("[data-testid='stFileUploader'] input[type='file']")
            await file_input.set_input_files(DUMMY_PDF)
            
            # Wait for upload/processing to start
            await asyncio.sleep(5) 
            
            # Capture status indicator updates
            status_screenshots = []
            start_time = time.time()
            while time.time() - start_time < 60: # Max 60s for build
                # Check for the presence of st.status or the build status indicator
                # We look for the text "문서 분석 중..." or similar
                content = await page.content()
                if "문서 분석 중" in content or "⏳" in content:
                    timestamp = int(time.time() * 1000)
                    screenshot_path = EVIDENCE_DIR / f"build_status_{timestamp}.png"
                    await page.screenshot(path=str(screenshot_path))
                    status_screenshots.append(screenshot_path)
                    await asyncio.sleep(3) # Check every 3s
                else:
                    # If not building, check if it finished
                    if len(status_screenshots) > 0:
                        break
                    await asyncio.sleep(2)
            
            print(f"  Captured {len(status_screenshots)} status screenshots.")

            await browser.close()

    except Exception as e:
        print(f"❌ An error occurred during QA: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Stopping Streamlit...")
        process.terminate()
        process.wait()
        print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(run_qa())
