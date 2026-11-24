"""Measure DOM lookups during a real Streamlit token stream.

Starts the deterministic autoscroll harness, wraps ``querySelector`` in the
top document, clicks ``append_stream``, and reports calls made by the real
production scroller JS while 60 chunks stream into the DOM.

Usage:
    python scripts/benchmarks/bench_stream_scroll.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests" / "e2e" / "_autoscroll_harness_app.py"
PORT = 8512
BASE_URL = f"http://127.0.0.1:{PORT}"
TRIALS = 3

SELECTORS = {
    '[data-testid="stChatInput"]',
    '[data-testid="stChatInput"] textarea',
    ':scope > [data-testid="stVerticalBlock"]',
    '[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]',
}

INSTALL_PROBE = """() => {
  const targets = new Set(__SELECTORS__);
  const state = {calls: []};
  const wrap = (proto) => {
    const orig = proto.querySelector;
    proto.querySelector = function (sel) {
      if (!targets.has(sel)) return orig.call(this, sel);
      const t0 = performance.now();
      const result = orig.call(this, sel);
      state.calls.push({sel, ms: performance.now() - t0});
      return result;
    };
  };
  wrap(Document.prototype);
  wrap(Element.prototype);
  for (const [name, proto] of [['getAttribute', Element.prototype], ['setAttribute', Element.prototype]]) {
    const orig = proto[name];
    proto[name] = function (key, value) {
      if (key !== 'aria-label') return orig.call(this, key, value);
      const t0 = performance.now();
      const result = orig.call(this, key, value);
      state.calls.push({sel: 'aria-label ' + name, ms: performance.now() - t0});
      return result;
    };
  }
  window.__scrollProbe = state;
  return true;
}""".replace("__SELECTORS__", json.dumps(sorted(SELECTORS)))

READ_PROBE = """() => {
  const calls = window.__scrollProbe ? window.__scrollProbe.calls : [];
  const bySel = {};
  for (const c of calls) {
    const row = bySel[c.sel] || (bySel[c.sel] = {count: 0, total_ms: 0});
    row.count += 1;
    row.total_ms += c.ms;
  }
  return {count: calls.length, total_ms: calls.reduce((a, c) => a + c.ms, 0), bySel};
}"""


def _wait_health() -> bool:
    for _ in range(10):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/_stcore/health", timeout=3) as r:
                if r.read().strip() == b"ok":
                    return True
        except OSError:
            time.sleep(1)
    return False


def _start_harness() -> subprocess.Popen[bytes] | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/_stcore/health", timeout=2) as r:
            if r.read().strip() == b"ok":
                return None
    except OSError:
        pass
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(HARNESS),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    if not _wait_health():
        proc.terminate()
        raise RuntimeError("autoscroll harness did not become healthy")
    return proc


def _one_trial(page: Any) -> dict[str, Any]:
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stChatInput"]', timeout=30000)
    page.wait_for_function(
        "() => !!window.__chatScrollState && !!window.__chatScrollState.observer",
        timeout=30000,
    )
    page.evaluate(INSTALL_PROBE)
    page.get_by_role("button", name="append_stream").click()
    page.wait_for_function(
        "() => document.body.innerText.includes('token-059 canned streaming chunk')",
        timeout=60000,
    )
    result = page.evaluate(READ_PROBE)
    page.evaluate("() => { delete window.__scrollProbe; }")
    return result


def main() -> None:
    proc = _start_harness()
    try:
        from playwright.sync_api import sync_playwright

        trials = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            try:
                for _ in range(TRIALS):
                    trials.append(_one_trial(page))
            finally:
                browser.close()
        result = {
            "tool": "bench_stream_scroll",
            "stream_chunks": 60,
            "trials": trials,
            "median_calls": statistics.median(t["count"] for t in trials),
            "median_total_ms": statistics.median(t["total_ms"] for t in trials),
            "per_trial": [
                {"count": t["count"], "total_ms": round(t["total_ms"], 3)}
                for t in trials
            ],
        }
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = reports / f"stream_scroll_bench_{stamp}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"wrote {path}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
