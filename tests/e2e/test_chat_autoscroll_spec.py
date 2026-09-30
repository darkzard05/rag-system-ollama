"""Playwright acceptance test for chat auto-scroll (red-light-first detector).

Covers the authoritative design document
``docs/superpowers/specs/2026-09-28-chat-autoscroll-playwright-visualqa-spec.md``
against the REAL production stack: the harness under ``tests/e2e/``
renders the REAL ``build_autoscroll_js`` / ``render_autoscroll_sentinel``
probe from ``src/ui/components/chat_autoscroll.py``, the REAL ``ui.py``
smart-scroller JS, and the REAL ``src/ui/styles/main.css``, then grows the
timeline with ``st.write_stream`` fed by a canned generator (no Ollama, no
PDF, no real question).

Assertion map (task contract -> spec section):

* A1 single scroller + ``window.scrollY === 0`` -> spec A1/A5 context
* A2 observer liveness + element identity -> spec A2 (+A5 probe slot)
* A3 no per-message micro-scrollbars -> spec G2
* A4 user-scroll holds, resume near bottom -> spec A3
* A5 delayed-observation auto-follow (CORE) -> spec A1/A4
* A7 sticky input invariant (delta <= 2px) -> spec S1/S2 accept criteria

HARD RULE: a precondition that cannot be established is a FAIL, never a
skip. This file contains no ``pytest.skip`` by design (a skip reported as
pass was the defect that let this feature fail three times).

Reuse: point ``AUTOSCROLL_BASE_URL`` (or ``STREAMLIT_URL``) at an
already-running harness, e.g.::

    streamlit run tests/e2e/_autoscroll_harness_app.py --server.port 8512
    AUTOSCROLL_BASE_URL=http://127.0.0.1:8512 pytest tests/e2e/ -q

Otherwise the session fixture spawns the harness itself on port 8512
(never 8510, which belongs to another agent).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import Any

import pytest
from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HARNESS_REL = os.path.join("tests", "e2e", "_autoscroll_harness_app.py")
HARNESS_PORT_DEFAULT = 8512

CHAT_INPUT_SEL = '[data-testid="stChatInput"]'
APPEND_BUTTON_NAME = "append_stream"
BOTTOM_THRESHOLD_PX = 8
NO_YANK_TOLERANCE_PX = 2
STICKY_DELTA_TOLERANCE_PX = 2
GROWTH_MIN_PX = 200

SCROLLER_SEL = (
    '[data-testid="stMainBlockContainer"] '
    '[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]'
)
SENTINEL_ID = "chat-autoscroll-sentinel"

FIND_CHAT_SCROLLER_JS = (
    "() => {"
    " const main = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
    " if (!main) return null;"
    " const cols = main.querySelectorAll('[data-testid=\"stColumn\"]');"
    " for (const col of cols) {"
    "  if (col.querySelector('[data-testid=\"stChatMessage\"]')) {"
    "   return col.querySelector(':scope > [data-testid=\"stVerticalBlock\"]');"
    "  }"
    " }"
    " return null;"
    "}"
)

METRICS_JS = (
    "() => {"
    " const scroller = (" + FIND_CHAT_SCROLLER_JS + ")();"
    " if (!scroller) return null;"
    " const cs = getComputedStyle(scroller);"
    " return {"
    "  scrollTop: scroller.scrollTop,"
    "  scrollHeight: scroller.scrollHeight,"
    "  clientHeight: scroller.clientHeight,"
    "  dist: scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight,"
    "  overflowY: cs.overflowY,"
    " };"
    "}"
)

DIAG_JS = (
    "() => {"
    " try {"
    "  const SEL = '" + SCROLLER_SEL + "';"
    "  const canonical = document.querySelector(SEL);"
    "  const anchored = (" + FIND_CHAT_SCROLLER_JS + ")();"
    "  const el = anchored || canonical;"
    "  const stt = window.__chatScrollState || null;"
    "  const sen = window.__chatSentinel || null;"
    "  const marker = document.getElementById('" + SENTINEL_ID + "');"
    "  return {"
    "   selector: SEL,"
    "   canonicalFound: canonical !== null,"
    "   anchoredFound: anchored !== null,"
    "   sameElement: canonical !== null && canonical === anchored,"
    "   scrollerNull: el === null,"
    "   scrollHeight: el ? el.scrollHeight : -1,"
    "   clientHeight: el ? el.clientHeight : -1,"
    "   scrollTop: el ? el.scrollTop : -1,"
    "   dist: el ? el.scrollHeight - el.scrollTop - el.clientHeight : -1,"
    "   overflowY: el ? getComputedStyle(el).overflowY : 'n/a',"
    "   chatScrollState: stt ? {"
    "    userScrolledUp: !!stt.userScrolledUp,"
    "    hasObserver: !!stt.observer,"
    "    isTicking: !!stt.isTicking,"
    "   } : 'missing',"
    "   chatSentinel: sen ? {"
    "    hasObserver: !!sen.observer,"
    "    wasVisible: !!sen.wasVisible,"
    "    ticking: !!sen.ticking,"
    "   } : 'missing',"
    "   sentinelMarkerPresent: marker !== null,"
    "   iframeCount: document.querySelectorAll('iframe').length,"
    "   windowScrollY: window.scrollY,"
    "  };"
    " } catch (e) { return { diagError: String(e) }; }"
    "}"
)


def _resolve_configured_url() -> str | None:
    """Return a reuse URL from env, or None when the fixture must spawn."""
    for name in ("AUTOSCROLL_BASE_URL", "STREAMLIT_URL"):
        val = os.environ.get(name, "").strip()
        if val:
            return val.rstrip("/")
    return None


def _wait_healthy(base_url: str, context: str) -> None:
    """Poll the Streamlit health endpoint until ready; FAIL on expiry."""
    deadline = time.time() + 60.0
    last_error = "unknown"
    health = base_url + "/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    return
                last_error = f"status={resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    pytest.fail(
        f"harness server never became healthy ({context}) at {base_url}: {last_error}",
        pytrace=False,
    )


@pytest.fixture(scope="session")
def harness_base_url() -> Generator[str, None, None]:
    """Provide the harness URL, spawning a server unless reuse is configured."""
    configured = _resolve_configured_url()
    if configured is not None:
        _wait_healthy(configured, context="reuse-url")
        yield configured
        return
    port = int(os.environ.get("AUTOSCROLL_PORT", str(HARNESS_PORT_DEFAULT)))
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        HARNESS_REL,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_healthy(base_url, context=f"spawned pid={proc.pid}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _diag_dump(page: Page) -> str:
    """Collect the actionable diagnostic dump from the live page."""
    raw: Any = await page.evaluate(DIAG_JS)
    if not isinstance(raw, dict):
        return f"diag-evaluate returned non-dict: {raw!r}"
    lines = ["--- autoscroll diagnostic dump ---"]
    for key in (
        "selector",
        "canonicalFound",
        "anchoredFound",
        "sameElement",
        "scrollerNull",
        "scrollHeight",
        "clientHeight",
        "scrollTop",
        "dist",
        "overflowY",
        "chatScrollState",
        "chatSentinel",
        "sentinelMarkerPresent",
        "iframeCount",
        "windowScrollY",
    ):
        lines.append(f"{key}={raw.get(key)!r}")
    if "diagError" in raw:
        lines.append(f"diagError={raw['diagError']!r}")
    lines.append("--- end dump ---")
    return "\n".join(lines)


async def _open_browser(pw: Playwright) -> Browser:
    """Launch headless Chromium at the spec viewport (1280x800)."""
    browser = await pw.chromium.launch(headless=True)
    return browser


async def _goto_ready(page: Page, base_url: str) -> None:
    """Open the harness and wait for the chat input; FAIL (never skip)."""
    try:
        await page.goto(base_url, timeout=15000)
    except PlaywrightError as exc:
        pytest.fail(
            f"harness server unreachable at {base_url}: {exc}",
            pytrace=False,
        )
    try:
        await page.wait_for_selector(CHAT_INPUT_SEL, timeout=60000)
    except PlaywrightTimeoutError as exc:
        dump = await _diag_dump(page)
        pytest.fail(
            f"stChatInput never appeared at {base_url}: {exc}\n{dump}",
            pytrace=False,
        )
    await page.wait_for_timeout(2000)


async def _require_metrics(page: Page) -> dict[str, Any]:
    """Read scroller metrics; FAIL with dump when the scroller is null."""
    metrics: Any = await page.evaluate(METRICS_JS)
    if not isinstance(metrics, dict):
        dump = await _diag_dump(page)
        pytest.fail(f"chat scroller not found (metrics null).\n{dump}", pytrace=False)
    return metrics


async def _require_overflow(page: Page) -> dict[str, Any]:
    """Require a scrollable chat column; FAIL (never skip) when flat."""
    metrics = await _require_metrics(page)
    if not metrics["scrollHeight"] > metrics["clientHeight"] + 50:
        dump = await _diag_dump(page)
        pytest.fail(
            "chat column did not overflow; scroll assertions are meaningless "
            f"without overflow: {metrics}\n{dump}",
            pytrace=False,
        )
    return metrics


async def _scroll_to_bottom(page: Page) -> dict[str, Any]:
    """Force the scroller to the bottom; FAIL when it will not stick."""
    await page.evaluate(
        "() => { const el = (" + FIND_CHAT_SCROLLER_JS + ")();"
        " if (el) el.scrollTop = el.scrollHeight; }"
    )
    await page.wait_for_timeout(300)
    metrics = await _require_metrics(page)
    if metrics["dist"] > BOTTOM_THRESHOLD_PX:
        dump = await _diag_dump(page)
        pytest.fail(
            "could not establish stuck-to-bottom precondition: "
            f"dist={metrics['dist']} > {BOTTOM_THRESHOLD_PX}\n{dump}",
            pytrace=False,
        )
    return metrics


async def _wait_stream_settled(page: Page, timeout_ms: int = 20000) -> None:
    """Wait until scrollHeight stops growing (stream fully flushed).

    Growth-onset alone fires early in the ~2s stream; asserting then would
    observe only a partial window. Settle = three consecutive identical
    scrollHeight readings spanning >= 750ms. FAIL (never skip) on timeout.
    """
    stable = 0
    last_height = -1
    deadline = timeout_ms / 1000.0
    budget = int(deadline * 4)
    for _ in range(budget):
        metrics = await _require_metrics(page)
        height = metrics["scrollHeight"]
        stable = stable + 1 if height == last_height else 0
        last_height = height
        if stable >= 3:
            return
        await page.wait_for_timeout(250)
    dump = await _diag_dump(page)
    pytest.fail(
        "stream never settled within "
        f"{timeout_ms}ms (scrollHeight kept growing or stalled mid-flush).\n"
        f"{dump}",
        pytrace=False,
    )


async def _read_user_scrolled_up(page: Page) -> bool | None:
    """Read the production wheel-guard flag; None when the state is missing."""
    val: Any = await page.evaluate(
        "() => window.__chatScrollState"
        " ? !!window.__chatScrollState.userScrolledUp : null"
    )
    return val if val is None or isinstance(val, bool) else None


async def _wheel_pulse(page: Page, dy: int) -> dict[str, Any]:
    """Send one wheel pulse and wait for the resulting scroll to land.

    Movement is awaited by condition (``scrollTop`` change); the trailing
    250ms only lets a glide animation finish so the follow-up read is stable.
    A pulse that moves nothing while room remains is itself evidence and is
    returned (not hidden) for the caller to record.
    """
    before = await _require_metrics(page)
    await page.mouse.wheel(0, dy)
    with contextlib.suppress(PlaywrightTimeoutError):
        await page.wait_for_function(
            "() => { const el = (" + FIND_CHAT_SCROLLER_JS + ")();"
            " return el !== null && el.scrollTop !== "
            + str(before["scrollTop"])
            + "; }",
            timeout=3000,
        )
    await page.wait_for_timeout(250)
    return await _require_metrics(page)


async def _move_mouse_to_scroller_center(page: Page) -> None:
    """Place the cursor over the chat scroller; FAIL with dump if absent.

    Mandatory after any action that moves the cursor (notably
    locator.click(), which leaves it over the clicked button — wheeling
    afterwards would scroll whatever is under the button instead of the
    chat column, freezing the observed scrollTop).
    """
    handle = await page.evaluate_handle(FIND_CHAT_SCROLLER_JS)
    try:
        element = handle.as_element()
        if element is None:
            dump = await _diag_dump(page)
            pytest.fail(f"scroller handle is not an element.\n{dump}", pytrace=False)
        box = await element.bounding_box()
        if box is None:
            dump = await _diag_dump(page)
            pytest.fail(f"scroller has no bounding box.\n{dump}", pytrace=False)
        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    finally:
        await handle.dispose()


async def _click_append_and_wait_growth(page: Page, prev_height: int) -> None:
    """Click append_stream and wait for scrollHeight growth; FAIL on stall."""
    await page.get_by_role("button", name=APPEND_BUTTON_NAME).click()
    growth_js = (
        "() => { const el = (" + FIND_CHAT_SCROLLER_JS + ")();"
        " return el !== null && el.scrollHeight > "
        + str(prev_height + GROWTH_MIN_PX)
        + "; }"
    )
    try:
        await page.wait_for_function(growth_js, timeout=20000)
    except PlaywrightTimeoutError:
        dump = await _diag_dump(page)
        pytest.fail(
            "append_stream produced no DOM growth within 20s: "
            f"prev scrollHeight={prev_height}\n{dump}",
            pytrace=False,
        )


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a1_single_scroller_and_window_locked(harness_base_url: str) -> None:
    """A1: exactly one scroll container in the chat column; window locked."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            res: Any = await page.evaluate(
                "() => {"
                " const main = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
                " if (!main) return null;"
                " const cols = Array.from(main.querySelectorAll('[data-testid=\"stColumn\"]'));"
                " const chatCols = cols.filter((c) =>"
                "  c.querySelector('[data-testid=\"stChatMessage\"]'));"
                " const chatCol = chatCols.length === 1 ? chatCols[0] : null;"
                " const scroller = chatCol ? chatCol.querySelector("
                "  ':scope > [data-testid=\"stVerticalBlock\"]') : null;"
                " const scrollerScrollable = scroller ? (function () {"
                "  const cs = getComputedStyle(scroller);"
                "  return (cs.overflowY === 'auto' || cs.overflowY === 'scroll')"
                "   && scroller.scrollHeight > scroller.clientHeight;"
                " })() : false;"
                " return {"
                "  chatColCount: chatCols.length,"
                "  scrollerFound: scroller !== null,"
                "  scrollerScrollable: scrollerScrollable,"
                "  windowScrollY: window.scrollY,"
                " };"
                "}"
            )
            if not isinstance(res, dict):
                dump = await _diag_dump(page)
                pytest.fail(f"A1 probe returned null.\n{dump}", pytrace=False)
            assert res["chatColCount"] == 1, (
                f"A1: expected exactly 1 chat column, got {res['chatColCount']}\n"
                f"{await _diag_dump(page)}"
            )
            assert res["scrollerFound"], (
                "A1: chat column stVerticalBlock must exist, "
                f"got {res}\n{await _diag_dump(page)}"
            )
            assert res["scrollerScrollable"], (
                "A1: chat column stVerticalBlock must be the single scroller, "
                f"got {res}\n{await _diag_dump(page)}"
            )
            assert res["windowScrollY"] == 0, (
                f"A1: main window must stay locked, scrollY={res['windowScrollY']}\n"
                f"{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a2_observer_liveness_and_element_identity(
    harness_base_url: str,
) -> None:
    """A2: canonical and anchored selectors agree; both observers are live."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            identity: Any = await page.evaluate(
                "() => {"
                " const canonical = document.querySelector('" + SCROLLER_SEL + "');"
                " const anchored = (" + FIND_CHAT_SCROLLER_JS + ")();"
                " return {"
                "  canonicalFound: canonical !== null,"
                "  anchoredFound: anchored !== null,"
                "  sameElement: canonical !== null && canonical === anchored,"
                " };"
                "}"
            )
            assert isinstance(identity, dict), (
                f"A2 identity probe failed: {identity!r}\n{await _diag_dump(page)}"
            )
            assert identity["canonicalFound"], (
                "A2: scroller must resolve via the canonical selector: "
                f"{identity}\n{await _diag_dump(page)}"
            )
            assert identity["anchoredFound"], (
                "A2: scroller must resolve via the chat-anchored search: "
                f"{identity}\n{await _diag_dump(page)}"
            )
            assert identity["sameElement"], (
                "A2: observer target must be the same live element the test "
                f"polls (detached-node observer is a failure mode): {identity}\n"
                f"{await _diag_dump(page)}"
            )
            try:
                await page.wait_for_function(
                    "() => window.__chatScrollState"
                    " && window.__chatScrollState.observer"
                    " && window.__chatSentinel"
                    " && window.__chatSentinel.observer",
                    timeout=10000,
                )
            except PlaywrightTimeoutError:
                dump = await _diag_dump(page)
                pytest.fail(
                    "A2: observer liveness not established within 10s "
                    "(smart-scroller MutationObserver and/or sentinel "
                    "IntersectionObserver missing — probe JS may never have "
                    f"run).\n{dump}",
                    pytrace=False,
                )
            marker: Any = await page.evaluate(
                "!!document.getElementById('" + SENTINEL_ID + "')"
            )
            assert marker is True, (
                "A2: #chat-autoscroll-sentinel marker must exist inside the "
                f"scroller.\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a3_no_per_message_micro_scrollbars(harness_base_url: str) -> None:
    """A3: no descendant of the scroller may scroll on its own."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            bad: Any = await page.evaluate(
                "() => {"
                " const scroller = (" + FIND_CHAT_SCROLLER_JS + ")();"
                " if (!scroller) return null;"
                " const offenders = [];"
                " for (const el of scroller.querySelectorAll('*')) {"
                "  const oy = getComputedStyle(el).overflowY;"
                "  if ((oy === 'auto' || oy === 'scroll')"
                "   && el.scrollHeight > el.clientHeight + 2) {"
                "   const cls = typeof el.className === 'string'"
                "    ? el.className.slice(0, 60) : '';"
                "   offenders.push(el.tagName + '/' + cls"
                "    + '/sh=' + el.scrollHeight + '/ch=' + el.clientHeight);"
                "   if (offenders.length >= 10) break;"
                "  }"
                " }"
                " return offenders;"
                "}"
            )
            if bad is None:
                dump = await _diag_dump(page)
                pytest.fail(f"A3 probe found no scroller.\n{dump}", pytrace=False)
            assert bad == [], (
                f"A3: {len(bad)} descendant(s) with their own scrollbar "
                f"(micro-scrollbars forbidden): {bad}\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a7_sticky_input_invariant(harness_base_url: str) -> None:
    """A7: the chat input wrapper stays pinned to the column bottom."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            pinned: Any = await page.evaluate(
                "() => {"
                " const scroller = (" + FIND_CHAT_SCROLLER_JS + ")();"
                " if (!scroller) return null;"
                " const input = scroller.querySelector('[data-testid=\"stChatInput\"]');"
                " if (!input) return null;"
                " const wrapper = input.parentElement;"
                " const sb = scroller.getBoundingClientRect();"
                " const wb = wrapper.getBoundingClientRect();"
                " return {"
                "  wrapperPosition: getComputedStyle(wrapper).position,"
                "  delta: wb.bottom - sb.bottom,"
                "  visible: wb.top < sb.bottom - 1 && wb.bottom > sb.top + 1,"
                " };"
                "}"
            )
            if not isinstance(pinned, dict):
                dump = await _diag_dump(page)
                pytest.fail(f"A7 probe found no input.\n{dump}", pytrace=False)
            assert pinned["wrapperPosition"] == "sticky", (
                "A7: input wrapper must be sticky, "
                f"got {pinned['wrapperPosition']}\n{await _diag_dump(page)}"
            )
            assert abs(pinned["delta"]) <= STICKY_DELTA_TOLERANCE_PX, (
                "A7: input must sit at the column bottom, "
                f"delta={pinned['delta']}\n{await _diag_dump(page)}"
            )
            assert pinned["visible"], (
                f"A7: input must be visible, got {pinned}\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a5_delayed_observation_autoscroll(harness_base_url: str) -> None:
    """A5 (CORE): appended stream content must pull a bottom-stuck view down.

    After ``append_stream`` grows the timeline, the scroller's ``scrollTop``
    must increase (or the scroller must sit at its bottom,
    dist <= BOTTOM_THRESHOLD_PX)
    within a bounded 5s window. No movement in that window is a FAIL.
    """
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            await _scroll_to_bottom(page)
            before = await _require_metrics(page)
            top0 = before["scrollTop"]
            sh0 = before["scrollHeight"]
            await _click_append_and_wait_growth(page, sh0)
            await _wait_stream_settled(page)
            after = await _require_metrics(page)
            followed = (
                after["scrollTop"] > top0 + NO_YANK_TOLERANCE_PX
                or after["dist"] <= BOTTOM_THRESHOLD_PX
            )
            assert followed, (
                "A5 CORE FAIL: scrollTop did not follow appended content within "
                "the 5s observation window — auto-scroll is dead on the real "
                "production JS path. "
                f"before(top={top0}, sh={sh0}) "
                f"after(top={after['scrollTop']}, sh={after['scrollHeight']}, "
                f"dist={after['dist']})\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_a4_user_scroll_holds_and_resumes(harness_base_url: str) -> None:
    """A4: scrolled-up view must not yank; return to bottom resumes follow."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            await _scroll_to_bottom(page)
            await _move_mouse_to_scroller_center(page)
            lifted = await _require_metrics(page)
            for _ in range(4):
                if lifted["dist"] > 300:
                    break
                lifted = await _wheel_pulse(page, -600)
            assert lifted["dist"] > 300, (
                "A4: could not lift the view >300px above bottom with real "
                f"wheel events (dist={lifted['dist']}).\n"
                f"{await _diag_dump(page)}"
            )
            flag_up = await _read_user_scrolled_up(page)
            assert flag_up is True, (
                "A4: wheel-up did not set userScrolledUp (wheel handler "
                "not attached? container re-created without "
                f"setupChatObserver re-run?). flag={flag_up}\n"
                f"{await _diag_dump(page)}"
            )
            top0 = lifted["scrollTop"]
            sh0 = lifted["scrollHeight"]
            await _click_append_and_wait_growth(page, sh0)
            await _wait_stream_settled(page)
            held = await _require_metrics(page)
            drift = abs(held["scrollTop"] - top0)
            assert drift <= NO_YANK_TOLERANCE_PX, (
                "A4 FAIL: view was yanked while scrolled up "
                f"(t0={top0} t1={held['scrollTop']} drift={drift}, "
                f"dist={held['dist']}).\n{await _diag_dump(page)}"
            )
            # The append click parks the cursor over the sidebar button;
            # re-seat it or resume wheels would scroll the sidebar, not chat.
            await _move_mouse_to_scroller_center(page)
            near = await _require_metrics(page)
            trace: list[str] = []
            for i in range(10):
                if near["dist"] <= BOTTOM_THRESHOLD_PX:
                    break
                near = await _wheel_pulse(page, 600)
                trace.append(f"pulse{i}: top={near['scrollTop']} dist={near['dist']}")
            assert near["dist"] <= BOTTOM_THRESHOLD_PX, (
                "A4: could not wheel back within BOTTOM_THRESHOLD_PX of the bottom "
                f"(dist={near['dist']}). wheel trace={trace}\n"
                f"{await _diag_dump(page)}"
            )
            flag_down = await _read_user_scrolled_up(page)
            assert flag_down is False, (
                "A4: wheeling down to within BOTTOM_THRESHOLD_PX did not clear "
                f"userScrolledUp. flag={flag_down}\n{await _diag_dump(page)}"
            )
            await _click_append_and_wait_growth(page, near["scrollHeight"])
            await _wait_stream_settled(page)
            final = await _require_metrics(page)
            assert final["dist"] <= BOTTOM_THRESHOLD_PX, (
                "A4 FAIL: auto-follow did not resume near the bottom "
                f"(dist={final['dist']}).\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_iframe_probe_slot_no_growth(harness_base_url: str) -> None:
    """Spec A5 analogue: probe iframes must occupy a deterministic slot."""
    async with async_playwright() as pw:
        browser = await _open_browser(pw)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1280, "height": 800})
            ).new_page()
            await _goto_ready(page, harness_base_url)
            await _require_overflow(page)
            count0: Any = await page.evaluate(
                "document.querySelectorAll('iframe').length"
            )
            before = await _require_metrics(page)
            await _click_append_and_wait_growth(page, before["scrollHeight"])
            # Count only once the stream has fully flushed: mid-stream the
            # sentinel probe slot churns transiently (observed 3 -> 2 -> 3).
            await _wait_stream_settled(page)
            count1: Any = await page.evaluate(
                "document.querySelectorAll('iframe').length"
            )
            assert count1 - count0 == 0, (
                "iframe probe slot grew across a rerun "
                f"({count0} -> {count1}); emission order is not "
                f"deterministic.\n{await _diag_dump(page)}"
            )
            assert count1 <= 4, (
                f"unexpected iframe population ({count1}); record the exact "
                f"selector used.\n{await _diag_dump(page)}"
            )
        finally:
            await browser.close()
