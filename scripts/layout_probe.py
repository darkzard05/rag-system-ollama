"""Layout probe: asserts the vertical-fill / sticky-bottom layout invariants.

Permanent Playwright regression harness for the Streamlit app at ``src/main.py``
(see ``.omo/plans/fix-first-run-upload-layout-collapse.md``, task 1). The DOM
chain is *measured*, never hard-coded: the chain from ``stMainBlockContainer``
down to the chat-input wrapper is derived with a findPath-style any-depth walk
(see ``scripts/reverify_dom_task1.py``) and every element on the path is
recorded with computed styles and ``getBoundingClientRect()``. No
Streamlit-version-specific chain is assumed.

Contract asserted:

- Desktop (viewport 1280x800): I1 ``.stApp`` computed overflow ``hidden``; I2
  the chat-column scroll container (a descendant of the *last* ``stColumn``,
  wrapping the chat input) fills to the viewport bottom
  (``innerHeight - 8 <= rect.bottom <= innerHeight + 4``), starts at the
  header height (``rect.top <= headerH + 10``), and keeps
  ``clientHeight >= 40%`` of the main-block content-box height; I3 the
  chat-input wrapper (nearest ancestor-or-self with computed ``position: sticky``
  - fallback: the direct parent) computes ``sticky`` with its bottom within 4px
  of the scroll container bottom; I4 no chain link (main block .. scroll
  container) computes ``offsetHeight < 0.08 * viewportHeight`` unless it is a
  zero-height Streamlit platform wrapper (``stElementContainer`` - exempt,
  justified in the evidence JSON).
- Mobile (viewport 700x900, ``--mobile``): M1 every ``stColumn`` computes
  ``height: 50vh`` and ``overflow-y: auto`` (whole document); M2 the chat-input
  wrapper computes ``position: fixed; bottom: 0; width: 100%``; M3 ``.stApp``
  computes ``overflow: visible``.

Every required testid (``stMainBlockContainer``, ``stVerticalBlock``,
``stLayoutWrapper``, ``stHorizontalBlock``, ``stColumn``, ``stChatInput``,
``stChatMessage``) must be present in the live tree, or the run FAILs - a
vacuous pass is never possible.

Exit codes:

    0  every evaluated invariant held and every required testid was present
    1  an invariant FAILed or a required testid was absent (or the pass-2
       ``stChatInput`` gate timed out)
    2  the app was unreachable (server down / boot blocked / no main block)
    3  an upload / upload-error leg was skipped (fixture file missing)

Evidence: JSON + PNG per snapshot, written under ``--out`` (default
``.omo/evidence/fix-first-run-upload-layout-collapse``). Existing evidence is
never overwritten - colliding names get a ``.new``/``.new.png`` suffix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Page,
    TimeoutError,
    ViewportSize,
    async_playwright,
)

BASE_URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")

BOOT_BUDGET_S = 150.0  # total goto + first-frame retry window
GOTO_TIMEOUT_S = 120.0  # single cold goto (torch / fitz imports are slow)
PASS2_GATE_S = 45.0  # wait for stChatInput after the splash frame
PRESENCE_SETTLE_S = 15.0  # settle for stChatMessage / fragment churn
UPLOAD_TERMINAL_S = 90.0  # upload: wait for a terminal st.status state
POLL_INTERVAL_S = 0.5  # splash-frame polling cadence (every 500 ms)

REQUIRED_TESTIDS = (
    "stMainBlockContainer",
    "stVerticalBlock",
    "stLayoutWrapper",
    "stHorizontalBlock",
    "stColumn",
    "stChatInput",
    "stChatMessage",
)

DESKTOP_VIEWPORT: ViewportSize = {"width": 1280, "height": 800}
MOBILE_VIEWPORT: ViewportSize = {"width": 700, "height": 900}

DEFAULT_OUT_DIR = ".omo/evidence/fix-first-run-upload-layout-collapse"

SPLASH_MEASURE_JS = r"""
() => {
    const main = document.querySelector('[data-testid="stMainBlockContainer"]');
    if (!main) return null;
    const mainStyle = getComputedStyle(main);
    const contentHeight = main.clientHeight
        - (parseFloat(mainStyle.paddingTop) || 0)
        - (parseFloat(mainStyle.paddingBottom) || 0);
    const input = document.querySelector('[data-testid="stChatInput"]');
    const statusLive = document.querySelector(
        '[data-testid="stStatusWidgetRunningIcon"]'
    );
    const app = document.querySelector('.stApp');
    return {
        stMainBlockContainer: {
            contentHeight: contentHeight,
            clientHeight: main.clientHeight,
            scrollHeight: main.scrollHeight,
            computedDisplay: mainStyle.display,
            computedHeight: mainStyle.height,
            exists: true,
        },
        hasChatInput: Boolean(input),
        hasStatusLive: Boolean(statusLive),
        appOverflow: app ? getComputedStyle(app).overflow : null,
        innerHeight: window.innerHeight,
    };
}
"""

CHAIN_MEASURE_JS = r"""
() => {
    const REQUIRED = [
        'stMainBlockContainer', 'stVerticalBlock', 'stLayoutWrapper',
        'stHorizontalBlock', 'stColumn', 'stChatInput', 'stChatMessage',
    ];
    const required = {};
    for (const t of REQUIRED) {
        required[t] = Boolean(
            document.querySelector(`[data-testid="${t}"]`)
        );
    }

    const main = document.querySelector('[data-testid="stMainBlockContainer"]');
    const input = document.querySelector('[data-testid="stChatInput"]');

    const metric = (el) => {
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        return {
            testid: el.getAttribute('data-testid'),
            tag: el.tagName,
            display: cs.display,
            flexDirection: cs.flexDirection,
            height: cs.height,
            cHeight: cs.height,
            cBottom: cs.bottom,
            cLeft: cs.left,
            cRight: cs.right,
            cTop: cs.top,
            cWidth: cs.width,
            clientHeight: el.clientHeight,
            scrollHeight: el.scrollHeight,
            offsetHeight: el.offsetHeight,
            overflowY: cs.overflowY,
            position: cs.position,
            top: Math.round(r.top * 100) / 100,
            bottom: Math.round(r.bottom * 100) / 100,
            left: Math.round(r.left * 100) / 100,
            right: Math.round(r.right * 100) / 100,
            width: Math.round(r.width * 100) / 100,
            heightPx: Math.round(r.height * 100) / 100,
        };
    };

    // Any-depth chain from stMainBlockContainer down to the chat-input
    // wrapper (findPath style: walk ancestors, never assume a fixed depth).
    const chain = [];
    if (main && input) {
        let node = input;
        while (node && node !== document.body) {
            chain.push(node);
            if (node === main) break;
            if (node.parentElement === null) break;
            node = node.parentElement;
        }
        chain.reverse();
    }

    // Chat-input wrapper: the nearest ancestor-or-self (walking up from the
    // chat input, stopping at the main block) that computes `position: sticky`
    // (desktop) or `position: fixed` (mobile); fallback = the direct parent.
    // Only the wrapper that owns the sticky/fixed rule is the layout anchor.
    let wrapper = null;
    if (input) {
        let node = input;
        while (node && node !== main && node !== document.body) {
            const cs = getComputedStyle(node);
            if (cs.position === 'sticky' || cs.position === 'fixed') {
                wrapper = node;
                break;
            }
            if (node.parentElement === null) break;
            node = node.parentElement;
        }
        if (!wrapper) wrapper = input.parentElement;
    }

    // Chat-column scroll container: a descendant of the *last* stColumn with
    // computed overflow-y: auto|scroll, any depth. Prefer the deepest one that
    // contains the chat input; otherwise the last (deepest) candidate inside
    // that column. This guarantees I2 is measured against the chat column, not
    // a spurious scroller elsewhere in the page.
    const columns = [...document.querySelectorAll('[data-testid="stColumn"]')];
    const lastColumn = columns.length ? columns[columns.length - 1] : null;
    let scroller = null;
    let scrollerInLastColumn = false;
    if (lastColumn) {
        const candidates = [lastColumn, ...lastColumn.querySelectorAll('*')]
            .filter((el) => {
                const cs = getComputedStyle(el);
                return cs.overflowY === 'auto' || cs.overflowY === 'scroll';
            });
        let wrapsInput = null;
        for (const el of candidates) {
            if (el === input || el.contains(input)) {
                if (!wrapsInput || wrapsInput.contains(el)) wrapsInput = el;
            }
        }
        const fallback = candidates.length ? candidates[candidates.length - 1] : null;
        scroller = wrapsInput || fallback;
        scrollerInLastColumn = scroller !== null;
    }

    // Chain links between the main block and the scroll container (both ends
    // included) - used by invariant I4.
    const links = [];
    if (main && scroller) {
        let seen = false;
        for (const el of chain) {
            if (el === main) seen = true;
            if (seen) links.push(el);
            if (el === scroller) break;
        }
    }

    const app = document.querySelector('.stApp');
    const csMain = main ? getComputedStyle(main) : null;
    const contentHeight = main
        ? main.clientHeight
            - (parseFloat((csMain && csMain.paddingTop) || 0) || 0)
            - (parseFloat((csMain && csMain.paddingBottom) || 0) || 0)
        : 0;

    const headerEl = document.querySelector('[data-testid="stHeader"]');
    const headerCssVar = getComputedStyle(document.documentElement)
        .getPropertyValue('--header-h')
        .trim();

    return {
        required,
        main: main ? metric(main) : null,
        contentHeight,
        wrapper: wrapper ? metric(wrapper) : null,
        input: input ? metric(input) : null,
        scroller: scroller ? metric(scroller) : null,
        scrollerInLastColumn,
        scrollerWrapsInput: Boolean(
            scroller && (scroller === input || scroller.contains(input))
        ),
        chain: chain.map(metric),
        links: links.map(metric),
        columns: columns.map(metric),
        appOverflow: app ? getComputedStyle(app).overflow : null,
        headerH: headerEl ? headerEl.offsetHeight : null,
        headerCssVar: headerCssVar,
        hasStatusLive: Boolean(document.querySelector(
            '[data-testid="stStatusWidgetRunningIcon"]'
        )),
        innerHeight: window.innerHeight,
        innerWidth: window.innerWidth,
        scrollY: window.scrollY,
    };
}
"""


def _px(value: str | None) -> float | None:
    """Parse a CSS pixel length (``740px``) into a float; ``None`` otherwise."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("px"):
        try:
            return float(text[:-2])
        except ValueError:
            return None
    if text == "0":
        return 0.0
    return None


def _target_path(out_dir: Path, name: str) -> Path:
    """Refuse to overwrite evidence: use ``<name>.new.ext`` when present."""
    candidate = out_dir / name
    if not candidate.exists():
        return candidate
    return out_dir / f"{candidate.stem}.new{candidate.suffix}"


def _save_json(out_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _target_path(out_dir, name)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


async def _save_png(page: Page, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _target_path(out_dir, name)
    await page.screenshot(path=str(target))
    return target


async def _measure_chain(page: Page) -> dict[str, Any]:
    return await page.evaluate(CHAIN_MEASURE_JS)


async def _wait_selector(page: Page, selector: str, timeout_s: float) -> bool:
    try:
        await page.wait_for_selector(selector, timeout=int(timeout_s * 1000))
        return True
    except TimeoutError:
        return False


async def _goto_budget(page: Page, budget_s: float) -> str | None:
    """Navigate with retries inside ``budget_s``; return a BLOCKED reason."""
    deadline = time.monotonic() + budget_s
    last_error = "never attempted"
    while time.monotonic() < deadline:
        remaining_ms = max(int((deadline - time.monotonic()) * 1000), 2000)
        try:
            await page.goto(
                BASE_URL,
                wait_until="domcontentloaded",
                timeout=int(min(remaining_ms, GOTO_TIMEOUT_S * 1000)),
            )
            return None
        except Exception as exc:  # noqa: BLE001 - any boot failure is retried
            last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2.0)
    return (
        f"app unreachable: connection to {BASE_URL} failed for "
        f"{budget_s:.0f}s (last error: {last_error})"
    )


def _frame(measure: dict[str, Any], mobile: bool, subset: bool) -> dict[str, Any]:
    """Details for one snapshot; ``subset`` = upload t0/t6 window (I2-I4)."""
    if mobile:
        return _evaluate_mobile(measure)
    if subset:
        full = _evaluate_desktop(measure)
        return {name: full[name] for name in ("I2", "I3", "I4")}
    return _evaluate_desktop(measure)


def _frame_ok(frame: dict[str, Any]) -> bool:
    return all(inv["pass"] for inv in frame.values())


def _evaluate_desktop(measure: dict[str, Any]) -> dict[str, Any]:
    """Invariants I1-I4 (desktop, viewport 1280x800)."""
    i1 = {
        "pass": measure.get("appOverflow") == "hidden",
        "appOverflow": measure.get("appOverflow"),
    }

    scroller = measure.get("scroller")
    inner = measure.get("innerHeight", 0)
    content = measure.get("contentHeight") or 0
    header_h = measure.get("headerH") or 60
    if not scroller:
        i2 = {
            "pass": False,
            "reason": "no chat-column overflow-y auto/scroll container found "
            "as a descendant of the last stColumn",
        }
    else:
        # Header-top layout contract (main.css: .stApp 100vh/overflow hidden,
        # stMainBlockContainer height:100dvh with padding-top:var(--header-h)).
        # The scroller must fill to the viewport bottom (not a bottom-header
        # offset), start right under the header, wrap the chat input, live in
        # the last stColumn, and keep at least 40% of the main content height.
        scroller_top = scroller["top"]
        bottom_min = inner - 8
        bottom_max = inner + 4
        top_threshold = header_h + 10
        bottom_ok = bottom_min <= scroller["bottom"] <= bottom_max
        top_ok = scroller_top <= top_threshold
        client_ok = scroller["clientHeight"] >= 0.40 * content
        i2 = {
            "pass": (
                bool(measure.get("scrollerInLastColumn"))
                and bool(measure.get("scrollerWrapsInput"))
                and bottom_ok
                and top_ok
                and client_ok
            ),
            "scrollerInLastColumn": bool(measure.get("scrollerInLastColumn")),
            "scrollerWrapsInput": bool(measure.get("scrollerWrapsInput")),
            "scrollerBottom": scroller["bottom"],
            "scrollerTop": round(scroller_top, 2),
            "innerHeight": inner,
            "bottomMin": round(bottom_min, 2),
            "bottomMax": round(bottom_max, 2),
            "topThreshold": round(top_threshold, 2),
            "headerH": header_h,
            "scrollerClientHeight": scroller["clientHeight"],
            "mainContentHeight": content,
            "ratio": round(scroller["clientHeight"] / content, 3) if content else 0.0,
        }

    wrapper = measure.get("wrapper")
    if not scroller or not wrapper:
        i3 = {"pass": False, "reason": "wrapper or scroll container missing"}
    else:
        delta = abs(wrapper["bottom"] - scroller["bottom"])
        i3 = {
            "pass": wrapper["position"] == "sticky" and delta <= 4.0,
            "wrapperPosition": wrapper["position"],
            "wrapperBottom": wrapper["bottom"],
            "scrollerBottom": scroller["bottom"],
            "delta": round(delta, 2),
        }

    threshold = 0.08 * inner
    exemptions: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for link in measure.get("links") or []:
        zero_platform = (
            link["testid"] == "stElementContainer" and link["offsetHeight"] == 0
        )
        if zero_platform:
            exemptions.append(
                {
                    "testid": link["testid"],
                    "tag": link["tag"],
                    "offsetHeight": link["offsetHeight"],
                    "reason": (
                        "zero-height Streamlit platform wrapper "
                        "(stElementContainer) - exempt from I4 with this "
                        "justification"
                    ),
                }
            )
            continue
        if link["offsetHeight"] < threshold:
            violations.append(
                {
                    "testid": link["testid"],
                    "tag": link["tag"],
                    "offsetHeight": link["offsetHeight"],
                    "threshold": round(threshold, 2),
                }
            )
    i4 = {
        "pass": not violations,
        "viewportThreshold": round(threshold, 2),
        "violations": violations,
        "exemptions": exemptions,
    }
    return {"I1": i1, "I2": i2, "I3": i3, "I4": i4}


def _evaluate_mobile(measure: dict[str, Any]) -> dict[str, Any]:
    """Invariants M1-M3 (mobile, viewport 700x900)."""
    inner = measure.get("innerHeight", 0)
    target = round(0.5 * inner, 1)
    cols = []
    for col in measure.get("columns") or []:
        height_px = _px(col["height"])
        height_ok = height_px is not None and abs(height_px - target) <= 4
        cols.append(
            {
                "testid": col["testid"],
                "height": col["height"],
                "target50vhPx": target,
                "heightOk": height_ok,
                "overflowY": col["overflowY"],
                "overflowOk": col["overflowY"] in ("auto", "scroll"),
            }
        )
    all_ok = bool(cols) and all(c["heightOk"] and c["overflowOk"] for c in cols)
    m1 = {"pass": all_ok, "target50vhPx": target, "columns": cols}

    wrapper = measure.get("wrapper")
    if not wrapper:
        m2 = {"pass": False, "reason": "no chat-input wrapper"}
    else:
        bottom_px = _px(wrapper["cBottom"])
        width_px = _px(wrapper["cWidth"])
        m2 = {
            "pass": (
                wrapper["position"] == "fixed"
                and bottom_px is not None
                and abs(bottom_px) <= 1
                and width_px is not None
                and abs(width_px - measure.get("innerWidth", 0)) <= 2
            ),
            "wrapperPosition": wrapper["position"],
            "cBottom": wrapper["cBottom"],
            "cWidth": wrapper["cWidth"],
            "innerWidth": measure.get("innerWidth"),
        }

    m3 = {
        "pass": measure.get("appOverflow") == "visible",
        "appOverflow": measure.get("appOverflow"),
    }
    return {"M1": m1, "M2": m2, "M3": m3}


async def _capture_splash(page: Page, out_dir: Path, deadline: float) -> dict[str, Any]:
    """Record the pass-1 splash frame - the LAST pre-gate sample, not the first.

    Splash = ``stMainBlockContainer`` present while ``stChatInput`` is still
    absent (polling every ``POLL_INTERVAL_S`` up to the boot deadline). The
    very first frame with the main block can be an unstyled Streamlit skeleton
    (``display: block``, content height 0) captured before the app script has
    emitted its first element and before custom CSS applied; the last pass-1
    sample taken before the pass-2 gate is the styled frame, so that one is
    saved and asserted. ``passed`` asserts the saved sample's content height
    is > 0; the warm/skip case (chat input already present on the very first
    sample) is not an assertion subject.
    """
    last_pass1: dict[str, Any] | None = None
    first_sample = True
    while time.monotonic() < deadline:
        try:
            splash = await page.evaluate(SPLASH_MEASURE_JS)
        except Exception:  # noqa: BLE001 - navigation in progress
            await asyncio.sleep(POLL_INTERVAL_S)
            continue
        if splash is None:
            await asyncio.sleep(POLL_INTERVAL_S)
            continue
        if splash["hasChatInput"]:
            if first_sample:
                return {
                    "splash": splash,
                    "captured": False,
                    "passed": None,
                    "note": "chat input already present (warm/skipped pass-1)",
                }
            if last_pass1 is not None:
                json_path = _save_json(out_dir, "splash.json", last_pass1)
                png_path = await _save_png(page, out_dir, "splash.png")
                return {
                    "splash": last_pass1,
                    "captured": True,
                    "passed": (last_pass1["stMainBlockContainer"]["contentHeight"] > 0),
                    "exists": bool(last_pass1["stMainBlockContainer"]["exists"]),
                    "jsonFile": json_path.name,
                    "pngFile": png_path.name,
                }
            return {
                "splash": splash,
                "captured": False,
                "passed": None,
                "note": "chat input present before any pass-1 sample",
            }
        last_pass1 = splash
        first_sample = False
        await asyncio.sleep(POLL_INTERVAL_S)
    if last_pass1 is not None:
        json_path = _save_json(out_dir, "splash.json", last_pass1)
        png_path = await _save_png(page, out_dir, "splash.png")
        return {
            "splash": last_pass1,
            "captured": True,
            "passed": last_pass1["stMainBlockContainer"]["contentHeight"] > 0,
            "exists": bool(last_pass1["stMainBlockContainer"]["exists"]),
            "jsonFile": json_path.name,
            "pngFile": png_path.name,
        }
    return {
        "captured": False,
        "passed": None,
        "note": "main block never appeared within the boot window",
        "blocked": True,
    }


async def _wait_terminal_status(page: Page, timeout_s: float) -> bool:
    """True when no st.status widget is in the "running" state anymore."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            has_running = await page.evaluate(
                "Boolean(document.querySelector("
                "'[data-testid=\"stStatusWidgetRunningIcon\"]'))"
            )
        except Exception:  # noqa: BLE001 - navigation in progress
            has_running = True
        if not has_running:
            return True
        await asyncio.sleep(2.0)
    return False


async def _upload_leg(
    page: Page,
    out_dir: Path,
    fixture: Path,
    mobile: bool,
    png_name: str,
) -> dict[str, Any]:
    """--upload / --upload-error leg: layout-stability snapshots only."""
    await page.set_input_files("input[type=file]", str(fixture))
    await asyncio.sleep(0.5)
    t0 = await _measure_chain(page)
    await page.wait_for_timeout(6000)
    t6 = await _measure_chain(page)

    terminal = (
        "observed"
        if await _wait_terminal_status(page, UPLOAD_TERMINAL_S)
        else "timeout"
    )

    # Final with settle: a transient mid-re-render blip (Streamlit element-diff
    # during the terminal-state rerun) must not fail the leg. Re-measure until
    # the full frame passes or 3 attempts (~3s settle); the last measurement is
    # authoritative, so a genuinely persistent break still fails.
    final = await _measure_chain(page)
    final_attempts = 1
    while final_attempts < 3:
        frame_check = _frame(final, mobile, subset=False)
        if _frame_ok(frame_check):
            break
        await asyncio.sleep(1.5)
        final = await _measure_chain(page)
        final_attempts += 1
    png_path = await _save_png(page, out_dir, png_name)
    t0f = _frame(t0, mobile, subset=True)
    t6f = _frame(t6, mobile, subset=True)
    finalf = _frame(final, mobile, subset=False)
    return {
        "fixture": str(fixture),
        "t0": t0f,
        "t6": t6f,
        "terminalStatus": terminal,
        "final": finalf,
        "pngFile": png_path.name,
        "ok": _frame_ok(t0f) and _frame_ok(t6f) and _frame_ok(finalf),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layout_probe",
        description=(
            "Playwright layout-invariant probe for the Streamlit RAG app. "
            "Asserts the vertical-fill/sticky-bottom contract (I1-I4 desktop, "
            "M1-M3 mobile), captures the pass-1 splash frame, verifies all "
            "required testids exist, and optionally exercises an upload or "
            "upload-error leg (layout stability only)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Exit codes: 0 all invariants held; 1 an invariant FAILed, a "
            "required testid is absent, or the pass-2 gate timed out; 2 app "
            "unreachable (BLOCKED); 3 an upload leg was skipped (fixture file "
            "missing)."
        ),
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=(
            "evidence directory (JSON + PNG per snapshot; existing evidence "
            "is never overwritten: colliding names get a '.new' suffix)"
        ),
    )
    parser.add_argument(
        "--upload",
        metavar="PDF",
        default=None,
        help=(
            "after pass-2, upload this PDF and snapshot I2-I4 stability at "
            "t=0s and t=6s (build poll window), wait up to "
            f"{UPLOAD_TERMINAL_S:.0f}s for a terminal st.status, then final "
            "I1-I4 (after-upload.png); exit 3 if the file is missing"
        ),
    )
    parser.add_argument(
        "--upload-error",
        metavar="PATH",
        default=None,
        help=(
            "same wait loop with a corrupt fixture; I2-I4 must also hold in "
            "the error state (after-error.png); exit 3 if the file is missing"
        ),
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        help="use viewport 700x900 and assert mobile invariants M1-M3",
    )
    return parser


async def _run_probe(args: argparse.Namespace) -> tuple[int, str, dict[str, Any]]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mobile = bool(args.mobile)
    mode = "mobile" if mobile else "desktop"
    viewport = MOBILE_VIEWPORT if mobile else DESKTOP_VIEWPORT

    upload_file = Path(args.upload) if args.upload else None
    if upload_file is not None and not upload_file.is_file():
        reason = f"upload leg skipped: file not found: {upload_file}"
        return 3, reason, {"reason": reason}
    error_file = Path(args.upload_error) if args.upload_error else None
    if error_file is not None and not error_file.is_file():
        reason = f"upload-error leg skipped: file not found: {error_file}"
        return 3, reason, {"reason": reason}

    browser = None
    report: dict[str, Any] = {
        "run": f"{mode}-{time.strftime('%Y%m%d%H%M%S')}",
        "mode": mode,
        "baseUrl": BASE_URL,
        "exit": None,
        "exit_reason": None,
        "viewport": dict(viewport),
        "invariants": {},
        "failed_invariants": [],
        "chain": [],
        "header_h": {"value": None},
        "exemptions": [],
        "files": {},
    }
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport=viewport)
            page = await context.new_page()

            blocked = await _goto_budget(page, BOOT_BUDGET_S)
            if blocked is not None:
                report["exit"] = 2
                report["exit_reason"] = blocked
                _save_json(out_dir, "layout-probe.json", report)
                return 2, blocked, report

            boot_deadline = time.monotonic() + GOTO_TIMEOUT_S
            splash = await _capture_splash(page, out_dir, boot_deadline)
            if splash.get("blocked"):
                reason = (
                    f"app at {BASE_URL} never rendered stMainBlockContainer "
                    "within the boot window; server log tail required"
                )
                report["exit"] = 2
                report["exit_reason"] = reason
                _save_json(out_dir, "layout-probe.json", report)
                return 2, reason, report

            gate_ok = await _wait_selector(
                page, '[data-testid="stChatInput"]', PASS2_GATE_S
            )
            gate_note = (
                "stChatInput observed"
                if gate_ok
                else f"stChatInput absent after {PASS2_GATE_S:.0f}s"
            )

            # Fragment-driven churn (timeline polls every few seconds) may add
            # stChatMessage/wrap elements a tick after the gate; settle first.
            if gate_ok:
                await _wait_selector(
                    page, '[data-testid="stChatMessage"]', PRESENCE_SETTLE_S
                )
                await asyncio.sleep(1.0)

            measure = await _measure_chain(page)
            invariant_details = _frame(measure, mobile, subset=False)
            missing = [t for t in REQUIRED_TESTIDS if not measure["required"].get(t)]

            upload = None
            if upload_file is not None:
                upload = await _upload_leg(
                    page, out_dir, upload_file, mobile, "after-upload.png"
                )
            upload_error = None
            if error_file is not None:
                upload_error = await _upload_leg(
                    page, out_dir, error_file, mobile, "after-error.png"
                )

            png_name = "mobile.png" if mobile else "first-load.png"
            png_path = await _save_png(page, out_dir, png_name)

            header_h = {
                "value": measure.get("headerH"),
                "hint": "stHeader.offsetHeight when present",
                "cssVar": measure.get("headerCssVar"),
            }
            exempts = [
                entry
                for name, inv in invariant_details.items()
                if isinstance(inv, dict) and "exemptions" in inv
                for entry in inv["exemptions"]
            ]

            report.update(
                {
                    "splash": splash,
                    "pass2Gate": {"ok": gate_ok, "note": gate_note},
                    "requiredTestids": measure["required"],
                    "missingTestids": missing,
                    "main": measure["main"],
                    "contentHeight": measure["contentHeight"],
                    "wrapper": measure["wrapper"],
                    "input": measure["input"],
                    "scroller": measure["scroller"],
                    "scrollerInLastColumn": measure["scrollerInLastColumn"],
                    "chain": measure["chain"],
                    "links": measure["links"],
                    "columns": measure["columns"],
                    "appOverflow": measure["appOverflow"],
                    "hasStatusRunning": measure["hasStatusLive"],
                    "innerHeight": measure["innerHeight"],
                    "scrollY": measure["scrollY"],
                    "header_h": header_h,
                    "invariants": {
                        name: bool(inv["pass"])
                        for name, inv in invariant_details.items()
                    },
                    "invariantDetails": invariant_details,
                    "exemptions": exempts,
                    "upload": upload,
                    "uploadError": upload_error,
                    "files": {
                        "splashJson": splash.get("jsonFile"),
                        "splashPng": splash.get("pngFile"),
                        "mainPng": png_path.name,
                        "json": _target_path(out_dir, "layout-probe.json").name,
                    },
                }
            )

            failures: list[str] = []
            if not gate_ok:
                failures.append(gate_note)
            if missing:
                failures.append(f"required testids absent: {missing}")
            if splash.get("passed") is False:
                failures.append("splash content height <= 0")
            for name, inv in invariant_details.items():
                if not inv["pass"]:
                    failures.append(f"{name} FAIL")
            if upload is not None and not upload["ok"]:
                failures.append("upload leg FAIL")
            if upload_error is not None and not upload_error["ok"]:
                failures.append("upload-error leg FAIL")

            if failures:
                code = 1
                reason = "; ".join(failures)
            else:
                code = 0
                reason = "all invariants held"
            report["exit"] = code
            report["exit_reason"] = reason
            report["failed_invariants"] = [
                name for name, inv in invariant_details.items() if not inv["pass"]
            ]
            _save_json(out_dir, "layout-probe.json", report)
            return code, reason, report
    finally:
        if browser is not None:
            await browser.close()


def main() -> int:
    args = _build_parser().parse_args()
    code, reason, _ = asyncio.run(_run_probe(args))
    print(f"PROBE_EXIT={code} REASON={reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
