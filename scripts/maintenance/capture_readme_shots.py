"""Capture the README screenshot from a real, running Streamlit app.

The script starts its own headless server, uploads the fixture PDF, asks one
fixed question, and writes ``<out>/image1.png`` plus a capture manifest.
A running Ollama with the configured models and Playwright Chromium are
required; no screenshot is produced without them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION = "이 문서의 핵심 내용을 요약해 주세요"
DEFAULT_PDF = "tests/data/2201.07520v1.pdf"
VIEWPORT: dict[str, int] = {"width": 1920, "height": 1080}
HEALTH_ATTEMPTS = 10

MESSAGE_COUNT_JS = (
    "(n) => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length > n"
)

BUILD_READY_JS = """
() => {
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  const uploaded = sidebar && (sidebar.innerText || "").includes(".pdf");
  const body = document.body.innerText || "";
  return Boolean(uploaded && /Ready|준비 완료|분석 완료/.test(body));
}
"""

VIEWER_READY_JS = """
() => {
  const ta = document.querySelector('[data-testid="stChatInput"] textarea');
  const viewer = [...document.querySelectorAll('iframe')].some((el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  });
  return Boolean(ta && !ta.disabled && viewer);
}
"""

ANSWER_DONE_JS = """
() => {
  const ta = document.querySelector('[data-testid="stChatInput"] textarea');
  const placeholder = ta ? (ta.placeholder || "") : "";
  const busyPlaceholder = /Generating|생성하고/.test(placeholder);
  const messages = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
  const live = messages.some((el) => {
    const text = el.innerText || "";
    return text.includes("▍") || text.includes("▌");
  });
  const last = messages.length
    ? (messages[messages.length - 1].innerText || "").trim()
    : "";
  return !busyPlaceholder && !live && last.length > 0;
}
"""

SNAPSHOT_JS = """
() => {
  const ta = document.querySelector('[data-testid="stChatInput"] textarea');
  const statuses = [...document.querySelectorAll('[data-testid="stStatusWidget"]')]
    .map((el) => (el.innerText || "").slice(0, 200));
  const messages = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
  return {
    placeholder: ta ? (ta.placeholder || "") : null,
    textareaDisabled: ta ? ta.disabled : null,
    statusTexts: statuses,
    runningStatusIcons: document.querySelectorAll(
      '[data-testid="stStatusWidgetRunningIcon"]'
    ).length,
    chatMessageCount: messages.length,
    lastMessageText: messages.length
      ? (messages[messages.length - 1].innerText || "").slice(0, 400)
      : "",
    bodyTail: (document.body.innerText || "").slice(-500),
  };
}
"""


class CaptureError(RuntimeError):
    """A screenshot capture could not be completed."""


@dataclass(frozen=True)
class CaptureConfig:
    """Typed CLI inputs, parsed once at the boundary."""

    out_dir: Path
    port: int
    question: str
    pdf_path: Path
    timeout_s: float


def _from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> CaptureConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="assets", help="output directory")
    parser.add_argument("--port", type=int, default=8517, help="Streamlit port")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="question sent to the chat input",
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF fixture to upload")
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="per-wait timeout in seconds"
    )
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        raise CaptureError(f"--port must be in 1..65535, got {args.port}")
    if args.timeout <= 0:
        raise CaptureError(f"--timeout must be positive, got {args.timeout}")

    pdf_path = _from_root(args.pdf)
    if not pdf_path.is_file():
        raise CaptureError(f"PDF fixture not found: {pdf_path}")
    return CaptureConfig(
        out_dir=_from_root(args.out),
        port=args.port,
        question=args.question,
        pdf_path=pdf_path,
        timeout_s=args.timeout,
    )


def _log_tail(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if path.exists():
            tail = path.read_bytes()[-2000:].decode("utf-8", errors="replace")
            parts.append(f"{path.name}: {tail}")
    return " | ".join(parts)


def _wait_for_health(
    port: int, proc: subprocess.Popen[bytes], log_paths: list[Path]
) -> None:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    last_error: OSError | None = None
    for attempt in range(1, HEALTH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                body = response.read().strip()
                if body == b"ok":
                    print(f"[capture] health ready after {attempt}s", flush=True)
                    return
                last_error = OSError(f"unexpected health body: {body[:80]!r}")
        except OSError as exc:
            last_error = exc
        if proc.poll() is not None:
            raise CaptureError(
                f"Streamlit exited early with code {proc.returncode}. "
                f"{_log_tail(log_paths)}"
            )
        if attempt < HEALTH_ATTEMPTS:
            time.sleep(1)
    raise CaptureError(
        f"Streamlit not healthy after {HEALTH_ATTEMPTS} attempts at {url}: "
        f"{last_error}. {_log_tail(log_paths)}"
    )


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    elif proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _new_temp_path(directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=".image1-", suffix=".png", dir=str(directory), delete=False
    ) as handle:
        return Path(handle.name)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_manifest(config: CaptureConfig) -> Path:
    payload: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "question": config.question,
        "viewport": VIEWPORT,
        "pdf_path": str(config.pdf_path),
        "port": config.port,
        "ollama_assumed_required": True,
    }
    target = config.out_dir / "capture_manifest.json"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json", dir=str(config.out_dir)
    )
    tmp = Path(tmp_name)
    moved = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, target)
        moved = True
    finally:
        if not moved:
            tmp.unlink(missing_ok=True)
    return target


async def _dom_snapshot(page: Page | None) -> dict[str, object]:
    if page is None or page.is_closed():
        return {"error": "browser page was never opened"}
    return await page.evaluate(SNAPSHOT_JS)


async def _capture(config: CaptureConfig, screenshot_tmp: Path) -> None:
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise CaptureError(
            "Playwright is required. Install it with: "
            "python -m pip install playwright && "
            "python -m playwright install chromium"
        ) from exc

    timeout_ms = int(config.timeout_s * 1000)
    url = f"http://127.0.0.1:{config.port}"
    phase = "app load"
    page: Page | None = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080}
                )
                page = await context.new_page()
                page.set_default_timeout(timeout_ms)
                await page.goto(url, wait_until="networkidle")
                await page.locator("[data-testid='stChatInput'] textarea").wait_for(
                    state="visible"
                )

                phase = "document build"
                await page.set_input_files(
                    "[data-testid='stSidebar'] input[type='file']",
                    str(config.pdf_path),
                )
                await page.wait_for_function(BUILD_READY_JS, timeout=timeout_ms)
                await page.wait_for_function(VIEWER_READY_JS, timeout=timeout_ms)

                phase = "answer stream"
                before = await page.locator("[data-testid='stChatMessage']").count()
                chat = page.locator("[data-testid='stChatInput'] textarea")
                await chat.fill(config.question)
                await chat.press("Enter")
                await page.wait_for_function(
                    MESSAGE_COUNT_JS, arg=before, timeout=timeout_ms
                )
                await page.wait_for_function(ANSWER_DONE_JS, timeout=timeout_ms)

                phase = "screenshot"
                await page.screenshot(path=str(screenshot_tmp), full_page=False)
            finally:
                await browser.close()
    except (PlaywrightTimeoutError, AssertionError) as exc:
        try:
            snapshot = await _dom_snapshot(page)
        except PlaywrightError as snapshot_exc:
            snapshot = {"error": f"DOM snapshot failed: {snapshot_exc}"}
        raise CaptureError(
            f"Timed out during {phase} after {config.timeout_s:g}s. "
            f"Observed DOM: {json.dumps(snapshot, ensure_ascii=False)}"
        ) from exc
    except PlaywrightError as exc:
        raise CaptureError(
            f"Browser automation failed during {phase}: {exc}. "
            "If Chromium is missing, run: python -m playwright install chromium"
        ) from exc


async def run(config: CaptureConfig) -> int:
    proc: subprocess.Popen[bytes] | None = None
    log_paths: list[Path] = []
    screenshot_tmp: Path | None = None
    try:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_fd, out_name = tempfile.mkstemp(prefix="readme-shot-", suffix=".stdout.log")
        err_fd, err_name = tempfile.mkstemp(prefix="readme-shot-", suffix=".stderr.log")
        log_paths = [Path(out_name), Path(err_name)]
        with os.fdopen(out_fd, "wb") as out_log, os.fdopen(err_fd, "wb") as err_log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "src/main.py",
                    "--server.port",
                    str(config.port),
                    "--server.headless",
                    "true",
                ],
                cwd=str(REPO_ROOT),
                stdout=out_log,
                stderr=err_log,
            )

        _wait_for_health(config.port, proc, log_paths)
        screenshot_tmp = _new_temp_path(config.out_dir)
        await _capture(config, screenshot_tmp)
        screenshot = config.out_dir / "image1.png"
        os.replace(screenshot_tmp, screenshot)
        screenshot_tmp = None
        manifest = _write_manifest(config)
        print(f"[capture] wrote {screenshot} and {manifest}", flush=True)
        return 0
    except (CaptureError, OSError) as exc:
        print(f"[capture] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if proc is not None:
            _terminate(proc)
        for path in log_paths:
            path.unlink(missing_ok=True)
        if screenshot_tmp is not None:
            screenshot_tmp.unlink(missing_ok=True)


def main() -> int:
    try:
        config = parse_args()
    except CaptureError as exc:
        print(f"[capture] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return asyncio.run(run(config))


if __name__ == "__main__":
    sys.exit(main())
