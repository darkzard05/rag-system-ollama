# SPEC: Chat Auto-Scroll — Playwright + Visual-QA Harness + Rollout / Do-Not-Touch Gate

> **SPEC ONLY. No `src/` edits. No `tests/` edits. No browser launch. No Ollama calls. No dependency changes.**
> QA must be able to execute this document **verbatim without Ollama** (mock `SessionManager` only).
> Stack pin: **Streamlit 1.60** (`requirements/base.txt` DOM contract; `main.css` header comment).

---

## 0. Ground truth (verified by Read/Grep — file:line)

| # | Fact | Source |
|---|------|--------|
| G1 | **L1–L3 flex chain** pins viewport height: L1 `.stApp { height: 100dvh; overflow: hidden }` → L2 `[data-testid="stMainBlockContainer"] { flex-direction: column; overflow: hidden; padding-top: var(--header-h) }` → L3 `[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { flex: 1; min-height: 0; flex-direction: column }` | `src/ui/styles/main.css:58`, `:60-72`, `:74-82` |
| G2 | **Single scroller**: the chat column's `stVerticalBlock` is the ONE scroll container (`overflow-y: auto`, `scrollbar-gutter: stable`); per-message wrappers must be natural height (`flex: 0 0 auto`, no `overflow-y:auto`) — no micro-scrollbars | `src/ui/styles/main.css:292-306`, `main.css:208-223`; enforced by `tests/e2e/test_chat_scroll.py:91-135` |
| G3 | **80px constant**: wheel-down resumes auto-scroll only when `dist = scrollHeight - scrollTop - clientHeight <= 80`; wheel-up (`deltaY < 0`) sets `userScrolledUp = true`; submit (Enter / send-button) clears the flag + forced `scrollToBottom(true)` after 50 ms; `MutationObserver(childList+subtree+characterData)` drives `scrollToBottom(false)` via `requestAnimationFrame` + `isTicking` guard; state lives at `window.parent.__chatScrollState` with observer disconnect-on-rerun | `src/ui/ui.py:125-208` (constant at `:162-163`); mirror in `dummy_autoscroll_test.py:116-123` |
| G4 | **Canonical scroller selector** (single-scroller resolution): `[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]` | `src/ui/ui.py:125-129`; e2e equivalent (chat-message-anchored) in `tests/e2e/test_chat_scroll.py:9-21` |
| G5 | **`st.iframe` precedent**: ALL auto-scroll/header JS is injected via `st.iframe(js, height="content")` (same-origin `window.parent.document` access; `st.html` sandbox would block it). Deterministic emission slot required — ghost `stLayoutWrapper` siblings break the `flex:1 :last-child` content-row rule | `src/ui/ui.py:73`, `:208`; `main.css:84-95`, `:106-118` |
| G6 | **STATUS — `jump_to_latest` key: NOT YET IN `src/`** (no match for `jump_to_latest` in `src/`; `src/ui/widget_keys.py` registry has `jump_key(msg_id,page,idx)` for reference page-jumps only, and the docstring states transient widget keys are intentionally NOT in `INTERACTIVE_KEYS`). This spec **defines** `jump_to_latest` as the rollout target (proposed widget key for the scroll-down affordance), it does not claim it exists | `src/ui/widget_keys.py:10-22`, `:43-51`, `:76-78`; `src/ui/components/chat.py:516-521` (`MAIN_CHAT_INPUT_KEY`, `submit_mode="stop"`) |
| G7 | **STATUS — `#chat-autoscroll-sentinel`: NOT YET IN `src/`** (no match in `src/`). This spec **defines** the sentinel contract (empty trailing div appended after the last message, before the sticky input wrapper) as the rollout target | `src/ui/components/chat.py:411-428` (`render_chat_messages_area` → `_render_unified_timeline`; insertion point for sentinel) |
| G8 | **Mock seam (no Ollama)**: `SessionManager` global fallback store + `set_ui_sync(None)` isolation; `DEFAULT_SESSION_STATE` keys (`messages`, `is_generating_answer`, `streaming_buffer`, …); `SessionManager.reset()` detaches UI-sync adapter | `src/core/session/manager.py:80-113`, `:138-148` |
| G9 | **Chat render order constraint (TASK5-a verdict)**: `render_chat_input_area()` BEFORE `render_chat_messages_area()` in `ui.py` (input widget must exist in DOM before the blocking `st.write_stream` loop); visual bottom-pin is `order:1` + `sticky bottom:0` + `margin-top:auto` in CSS | `src/ui/ui.py:233-253`; `main.css:320-352` |
| G10 | **CI gates (exact)**: `ruff check .` / `ruff format --check .` / `mypy src` (PYTHONPATH=src) / `bandit -r src/ -ll` / `pip-audit …` / `python scripts/maintenance/run_ci_checks.py` → unit: `python -m pytest --cov=src --cov-report=term --cov-report=xml --disable-warnings --cov-fail-under=55 tests/unit/`; integration: per-file `python -m pytest <file>` for the 14 files in `INTEGRATION_FILES` | `.github/workflows/ci.yml:42-48`, `:70-82`, `:120-123`; `scripts/maintenance/run_ci_checks.py:30-49`, `:69-83` |

### Implication for this spec

The harness below gates the **rollout** that introduces exactly two DOM additions and nothing else:

1. `#chat-autoscroll-sentinel` — empty trailing `<div id="chat-autoscroll-sentinel">` as the last message-flow child of the chat scroller (before the sticky input wrapper), observed via `IntersectionObserver` as the primary bottom-proximity signal, with the existing `dist <= 80` math retained as fallback.
2. `jump_to_latest` — `st.button("↓ 최신으로", key="jump_to_latest")` affordance, visible only when `userScrolledUp == true`, `on_click` clears the flag + forced scroll.

Until that rollout lands, assertions A2/A4/A5 run against the **fallback path** (see each assertion's `Pre-rollout` note) and MUST still pass.

---

## 1. Playwright assertions (5) — selectors + tolerances

Shared prelude for every assertion (paste at top of the spec test file when it is later created):

```python
import os
import pytest
from playwright.async_api import async_playwright

BASE_URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")

SCROLLER_SEL = (
    '[data-testid="stMainBlockContainer"] '
    '[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]'
)
SENTINEL_SEL = "#chat-autoscroll-sentinel"
JUMP_BTN_SEL = '[data-testid="stButton"] button[kind="secondary"]'  # key=jump_to_latest
# Tighter alternative when Streamlit exposes the key in the iframe DOM:
# JUMP_BTN_KEY_SEL = 'button[data-testid="baseButton-secondary"]'

SCROLL_STATE_JS = "() => window.parent.__chatScrollState || window.__chatScrollState || null"
METRICS_JS = f"""() => {{
    const c = document.querySelector('{SCROLLER_SEL}');
    if (!c) return null;
    return {{
        scrollTop: c.scrollTop, scrollHeight: c.scrollHeight,
        clientHeight: c.clientHeight,
        dist: c.scrollHeight - c.scrollTop - c.clientHeight,
        overflowY: getComputedStyle(c).overflowY,
    }};
}}"""
```

Setup block (each test): launch headless Chromium → `page.goto(BASE_URL, timeout=15000)` (on failure `pytest.skip`, same pattern as `tests/e2e/test_chat_scroll.py:70-74`) → `page.wait_for_selector('[data-testid="stChatInput"]', timeout=60000)` → `page.wait_for_timeout(2000)` settle → inject 50-paragraph overflow div **before** the sticky input wrapper (same insertion position as `tests/e2e/test_chat_scroll.py:152-179`).

### A1 — bottom proximity ≤ 80px after streaming append (tolerance: dist ≤ 80, rAF settle 500 ms)

- **Intent**: while stuck to bottom, a `MutationObserver`-observed DOM append keeps the viewport within the 80px constant of the bottom.
- **Steps**:
  1. `scrollTop = scrollHeight` (force to bottom), `wait_for_timeout(200)`.
  2. Append one `<p>Test scroll content line. ×8</p>` node before the input wrapper (mirrors streaming token chunk).
  3. `wait_for_timeout(500)` (rAF + `MutationObserver` + `isTicking` guard, `ui.py:138-144`, `:171-179`).
  4. Read `METRICS_JS` → `dist`.
- **Expect**: `dist <= 80`. **Tolerance**: `+0px` (hard gate on the `ui.py:163` constant); timing tolerance 500 ms settle, retry once after +500 ms before FAIL.
- **Pre-rollout**: passes via current `MutationObserver → scrollToBottom(false)` path (`ui.py:171-179`).
- **FAIL example**: `dist=412 > 80 after 2×500ms — observer disconnected on rerun?`.

### A2 — sentinel visibility at bottom (tolerance: `isIntersecting == true`, ratio > 0)

- **Intent**: the sentinel is the primary bottom signal post-rollout.
- **Steps**:
  1. Scroll to bottom. 2. `sentinel = page.wait_for_selector(SENTINEL_SEL, timeout=5000)`.
  3. `visible = page.evaluate("(s) => { const r = document.querySelector(s).getBoundingClientRect(); const c = document.querySelector(SCROLLER).getBoundingClientRect(); return r.bottom <= c.bottom + 2 && r.bottom >= c.bottom - 82; }")`.
- **Expect**: sentinel exists; `isIntersecting` (via `IntersectionObserver` with `root` = scroller) `== true`; bounding-rect tolerance **±2px** on the bottom edge, **80px** upper band (mirrors the 80px constant).
- **Pre-rollout**: sentinel absent → record `SKIP(sentinal-missing, fallback-path)` and assert fallback instead: `dist <= 80` (same as A1). SKIP here is a PASS for the gate (see §4).
- **FAIL example**: `sentinel present but isIntersecting=false at scrollHeight — sentinel placed outside scroller?`.

### A3 — no-yank when scrolled up > 300px (`scrollTop` unchanged, tolerance ±2px)

- **Intent**: reading history must never be yanked by streaming appends (`userScrolledUp` guard, `ui.py:136`).
- **Steps**:
  1. Scroll to bottom, record `sh`. 2. `mouse.wheel(0, -600)` over the scroller center (fires the real `wheel` handler; synthetic `scrollTop=` assignment does NOT set the flag — `ui.py:156-168` listens to `wheel` only).
  3. Verify `sh - scrollTop - clientHeight > 300` (else inject more content and repeat).
  4. Record `t0 = scrollTop`. 5. Append 5 paragraphs. 6. `wait_for_timeout(500)`. 7. Record `t1 = scrollTop`.
- **Expect**: `|t1 - t0| <= 2` (sub-pixel/rAF tolerance) AND `jump_to_latest` button visible (post-rollout) — visibility check is non-blocking pre-rollout.
- **Pre-rollout**: `__chatScrollState.userScrolledUp == true` asserted via `SCROLL_STATE_JS`.
- **FAIL example**: `t0=812 t1=1404 — wheel handler not attached (container re-created on rerun without setupChatObserver re-run?)`.

### A4 — fallback target when sentinel removed (tolerance: dist ≤ 80, sentinel query returns null)

- **Intent**: robustness — deleting the sentinel must not break auto-scroll; the `dist <= 80` math path takes over.
- **Steps**:
  1. Scroll to bottom. 2. `page.evaluate("document.querySelector('#chat-autoscroll-sentinel')?.remove()")` (post-rollout; pre-rollout this is a no-op — sentinel already absent).
  3. Append one paragraph. 4. `wait_for_timeout(500)`. 5. Read `METRICS_JS`.
- **Expect**: sentinel query returns `null` AND `dist <= 80`.
- **FAIL example**: `dist=233 after sentinel removal — fallback getChatContainer() returned null?`.

### A5 — iframe singleton ≤ 1 across 3 reruns (tolerance: count ≤ 1, zero growth)

- **Intent**: `st.iframe` probe injection must occupy a deterministic slot — duplicate iframes (ghost `stLayoutWrapper` siblings) collapse the content row (`main.css:84-118` BUG SHIELD).
- **Steps**: 3× (`page.reload()` → wait for `stChatInput` → settle 2000 ms) → each time count `document.querySelectorAll('iframe[title="st.iframe"]')` … note: Streamlit 1.60 titles iframes by component; if title varies, count `document.querySelectorAll('[data-testid="stIFrame"] iframe, iframe[srcdoc]')` and record the exact selector used.
- **Expect**: count `<= 1` on every run AND `count_run3 - count_run1 == 0` (no growth).
- **FAIL example**: `iframe count 1 → 2 → 3 across reruns — probe not in deterministic slot (re-read ui.py:82-208 emission order)`.

---

## 2. Screenshots (3) — viewport, naming, acceptance

- Viewport: **1280×800** (matches `scripts/layout_probe.py` `DESKTOP_VIEWPORT`), `device_scale_factor=1`.
- Output dir: `reports/visual_qa_autoscroll/` (repo-ignored artifacts dir; never commit PNGs — commit the spec + PASS/FAIL table only).
- Command pattern (verbatim, no Ollama involved):
  `STREAMLIT_URL=http://127.0.0.1:8501 python -m pytest tests/e2e/test_chat_autoscroll_spec.py -v` (file to be created at rollout; until then screenshots are captured via the manual run in §3 + DevTools).

| Shot | Setup | File | Must show (accept) |
|------|-------|------|--------------------|
| S1 `bottom-stuck` | mock session with 30 messages (§3 step 4), scrolled to bottom, streaming mid-chunk | `s1_bottom_stuck.png` | last message fully visible; `stChatInput` sticky bottom, `delta ≤ 2px`; no per-message scrollbar |
| S2 `scrolled-up+button` | `mouse.wheel(0,-600)` from bottom (>300px up) | `s2_scrolled_up_button.png` | `jump_to_latest` ("↓ 최신으로") button visible bottom-right of chat column (post-rollout); pre-rollout: history message visible + input still pinned (button absence noted, not a FAIL) |
| S3 `streaming-mid-chunk` | during `st.write_stream` mock token loop (§3 step 5) | `s3_streaming_mid_chunk.png` | `▌` cursor / partial token text visible AND viewport still at bottom (`dist ≤ 80`) — proves observer fires per-chunk, not only per-message |

---

## 3. Manual run — `streamlit run src/main.py` with mock session (no Ollama)

Exact commands (verbatim):

```powershell
# 0. Prerequisites (no Ollama daemon needed; app must boot without models for UI QA)
$env:PYTHONPATH = "src"
streamlit run src/main.py --server.port 8501 --server.headless true
# Open http://127.0.0.1:8501 in Chromium/Edge
```

Mock-session steps (all in-browser, zero Ollama calls):

1. **Boot check**: sidebar loads, `global_status` shows "문서 업로드 대기 중" (`manager.py:97`) — confirms `SessionManager` fallback store path, no RAG build triggered. **Do NOT upload a PDF, do NOT submit a question** (either would attempt model load).
2. **Seed history without backend**: open DevTools Console on the Streamlit page and inject overflow content into the chat scroller (mirrors `test_chat_scroll.py:152-179`):
   ```js
   const c = document.querySelector('[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]');
   const d = document.createElement('div'); d.style.padding = '10px';
   d.innerHTML = Array(50).fill('<p style="padding:8px;border-bottom:1px solid #eee;">' + 'Test scroll content line. '.repeat(8) + '</p>').join('');
   const w = c.querySelector(':scope > [data-testid="stElementContainer"]:has(> [data-testid="stChatInput"])');
   w ? c.insertBefore(d, w) : c.appendChild(d);
   ```
3. **A3 manual**: scroll to bottom → wheel up hard (`Shift+wheel` or 3 full notches) → confirm in Console `window.parent.__chatScrollState.userScrolledUp === true` → append one more paragraph via Console → confirm `scrollTop` unchanged (±2px).
4. **A1/A4 manual**: `c.scrollTop = c.scrollHeight` → confirm `userScrolledUp === false` after wheeling down to within 80px → append paragraph → confirm viewport sticks (`scrollHeight - scrollTop - clientHeight <= 80`).
5. **S3 manual**: use the repo's zero-Ollama streaming mock — separate terminal, `streamlit run dummy_autoscroll_test.py --server.port 8502`, check "스마트 자동 스크롤(Fix) 활성화", submit 2–3 questions in sequence, screenshot mid-`write_stream` (`mock_stream_tokens`, 40 ms/word). This exercises the identical `MutationObserver + rAF + 80px` JS (`dummy_autoscroll_test.py:51-163`) without touching `src/`.
6. **A5 manual**: reload the `src/main.py` page 3× → DevTools Elements → count `iframe` nodes → must be ≤ 1 and non-growing.

Record each step PASS/FAIL in the §4 table. Total manual time budget: **≤ 15 min**.

---

## 4. Consolidated phase-gate table (per-phase PASS/FAIL — rollout blocked on any FAIL)

| Phase | Gate | Command / procedure | PASS criteria | Result |
|-------|------|---------------------|---------------|--------|
| P0 Static | `ruff check .` | `ruff check .` | exit 0 | ☐ PASS / ☐ FAIL |
| P0 Static | `ruff format --check .` | `ruff format --check .` | exit 0 | ☐ PASS / ☐ FAIL |
| P0 Static | `mypy src` | `$env:PYTHONPATH="src"; mypy src` | exit 0, no errors on changed files | ☐ PASS / ☐ FAIL |
| P1 Unit+coverage | unit gate (unchanged src must stay green) | `python -m pytest --cov=src --cov-report=term --cov-fail-under=55 tests/unit/` | exit 0, coverage ≥ 55% | ☐ PASS / ☐ FAIL |
| P1 Integ | 14-file integration list unchanged | `python scripts/maintenance/run_ci_checks.py` (runs `run_scripts_import_sweep` + 14 `INTEGRATION_FILES` + unit+coverage) | exit 0 | ☐ PASS / ☐ FAIL |
| P2 Playwright | A1 bottom proximity | `STREAMLIT_URL=http://127.0.0.1:8501 python -m pytest tests/e2e/test_chat_autoscroll_spec.py::test_bottom_proximity -v` (post-rollout file) | `dist ≤ 80` | ☐ PASS / ☐ FAIL |
| P2 Playwright | A2 sentinel visibility | `…::test_sentinel_visibility -v` | intersecting (or SKIP+fallback PASS pre-rollout) | ☐ PASS / ☐ FAIL / ☐ SKIP→PASS |
| P2 Playwright | A3 no-yank scrolled-up | `…::test_no_yank_scrolled_up -v` | `\|ΔscrollTop\| ≤ 2` with >300px lift | ☐ PASS / ☐ FAIL |
| P2 Playwright | A4 sentinel-removed fallback | `…::test_fallback_sentinel_removed -v` | sentinel null + `dist ≤ 80` | ☐ PASS / ☐ FAIL |
| P2 Playwright | A5 iframe singleton ×3 | `…::test_iframe_singleton -v` | count ≤ 1, zero growth | ☐ PASS / ☐ FAIL |
| P3 Visual | S1 bottom-stuck | manual §3 / DevTools screenshot | accept criteria §2 | ☐ PASS / ☐ FAIL |
| P3 Visual | S2 scrolled-up+button | manual §3 / DevTools screenshot | accept criteria §2 | ☐ PASS / ☐ FAIL |
| P3 Visual | S3 streaming mid-chunk | `streamlit run dummy_autoscroll_test.py --server.port 8502` + screenshot | accept criteria §2 | ☐ PASS / ☐ FAIL |
| P4 Rollout | Do-Not-Touch audit (§5) | `git diff --stat -- <each protected path>` empty | zero diff on all §5 paths | ☐ PASS / ☐ FAIL |

**Gate rule**: P0–P1 must be green before any P2 run (a red P0/P1 invalidates P2 evidence). Any single FAIL blocks merge; SKIP is allowed only for A2-pre-rollout and S2-button-absent, each requiring the documented fallback PASS.

---

## 5. Do-Not-Touch list (rollout must not modify)

| Path | Why (regression it guards) |
|------|----------------------------|
| `src/ui/styles/main.css:58-135` (L1–L3 chain + content-row `:last-child` BUG SHIELD) | viewport collapse 840→420; ghost-wrapper 50:50 split |
| `src/ui/styles/main.css:292-352` (single-scroller + sticky input `order:1`) | micro-scrollbars; input unpin mid-scroll |
| `src/ui/ui.py:32-73` (`inject_header_height_script`, 60px fallback) | header overlap / `--header-h` drift (C5) |
| `src/ui/ui.py:211-253` (`render_main_content` column ratio 42/58 + input-before-messages TASK5-a order) | streaming input disappearance; Tab-order contract |
| `src/ui/widget_keys.py` (`INTERACTIVE_KEYS`, `MAIN_CHAT_INPUT_KEY`, `PDF_NAV_INPUT_KEY`) | widget-state flicker; adding transient keys resurrects stale clicks |
| `src/ui/components/chat.py:473-521` (`render_chat_input_area`, `submit_mode="stop"`, `MAIN_CHAT_INPUT_KEY`) | stop-button loss; INT-입력동결 regression |
| `src/core/session/manager.py` (store semantics, `reset()` isolation) | test isolation; cross-session bleed |
| `config.yml` (models, chunk sizes, reranker) | retrieval quality drift; Ollama model pull churn |
| `requirements*.txt`, `pyproject.toml` | dependency changes forbidden by this task |
| `tests/e2e/test_chat_scroll.py` (existing 9-check suite) | must keep passing unmodified alongside the new spec file |

**Allowed to change at rollout** (and only these): append sentinel `<div>` in the timeline render path (`chat.py` unified timeline, after last message / before input wrapper); add `jump_to_latest` button + its `on_click` in the chat column; extend the `st.iframe` scroller JS in `ui.py:94-208` with `IntersectionObserver` on the sentinel + button visibility toggle. The `dist <= 80` constant, selector, and `st.iframe` mechanism stay byte-identical.

---

## 6. Atomic commit plan (spec commit — this document only)

```powershell
# Single atomic commit; spec doc only — verifies zero src/tests drift by construction
git status --porcelain
git add docs/superpowers/specs/2026-09-28-chat-autoscroll-playwright-visualqa-spec.md
git diff --cached --stat
# Expected: 1 file, +N/-0; src/ tests/ config.yml requirements* untouched
git commit -m "spec(chat-scroll): Playwright+visual-QA harness, phase gates, Do-Not-Touch list"
git log --oneline -3
git show --stat HEAD
# Rollback (if gate fails before push): git reset --soft HEAD~1
```

Post-spec rollout (separate future commit, NOT this task): implement sentinel + `jump_to_latest` → run §4 table top-to-bottom → attach `reports/visual_qa_autoscroll/` PASS table → merge only on all-green.

---

## 7. Assumptions recorded

1. `jump_to_latest` and `#chat-autoscroll-sentinel` do not exist in `src/` today (verified by grep); the spec defines them as rollout targets with fallback-path acceptance so QA is executable now.
2. `dummy_autoscroll_test.py` (root) is accepted as the zero-Ollama streaming mock because it embeds the identical scroller JS and real `main.css`; it is NOT part of `src/` and needs no edits.
3. Playwright selector for the jump button uses the generic `stButton` secondary-button selector with a key-specific alternative noted; the exact `data-testid` key exposure is confirmed at rollout without changing this spec's tolerances.
4. Screenshots are evidence artifacts in `reports/` (git-ignored) and are never committed; the committed record is the spec + phase-gate table.
