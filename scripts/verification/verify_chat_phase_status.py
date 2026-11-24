"""Verify the completed chat status line on the running Streamlit app.

Usage:
    python scripts/verification/verify_chat_phase_status.py

The app must already serve on ``--url`` and Ollama must be reachable. The
check uploads the bundled PDF, asks one question, and asserts that the final
status caption marks all three answer phases as complete.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "tests" / "data" / "2201.07520v1.pdf"
QUESTION = "What is CM3? Answer briefly."


def wait_for_text(page, needle: str, timeout_s: int) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = " ".join(page.locator("body").inner_text().split())
        if needle in text:
            return text
        page.wait_for_timeout(500)
    raise TimeoutError(f"{needle!r} not visible after {timeout_s}s")


def wait_for_final_caption(page, timeout_s: int) -> str:
    deadline = time.monotonic() + timeout_s
    message = page.locator('[data-testid="stChatMessage"]').last
    caption = message.locator('[data-testid="stCaptionContainer"] p').first
    while time.monotonic() < deadline:
        line = caption.inner_text()
        if "Citations" in line and "▍" not in line:
            return line
        page.wait_for_timeout(500)
    raise TimeoutError(f"final status caption not visible after {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--upload-timeout", type=int, default=90)
    parser.add_argument("--answer-timeout", type=int, default=120)
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            box = page.locator('[data-testid="stChatInput"] textarea')
            box.wait_for(state="visible", timeout=60_000)
            page.locator('input[type="file"]').set_input_files(str(args.pdf.resolve()))
            wait_for_text(page, "Ready", args.upload_timeout)

            box.fill(QUESTION)
            box.press("Enter")
            wait_for_text(page, "Citations", args.answer_timeout)
            line = wait_for_final_caption(page, 30)
            checks = line.count("✓")
            print(f"final status: {line}")
            print(f"completed phases: {checks}/3")
            if checks != 3:
                print("FAIL: final status does not mark all three phases")
                return 1
            print("PASS: final status marks all three phases")
            return 0
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
