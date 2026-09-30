# Wave-0 Task 4 — Fragment-warning emitter attribution (STATIC-PARTIAL)

- Date (UTC+9): 2026-09-26 14:29 | Streamlit 1.60.0 | Repo `C:/Users/darkzard05/hy/rag-system-ollama`
- Target warning: `The fragment with id ... does not exist anymore - it might have been removed during a preceding full-app rerun.`
- Mode: **STATIC-PARTIAL** — live headless rerun attempted once (see §4); the timer-race path needs a real
  browser session, so no live stack was obtainable. No stack line from the app was observed; per the task rule
  (§5) the app-side attribution below is evidence-graded, not claimed from a stack.
- Fix status: **No fix applied** (Task 9 lane). `src/` read-only; nothing committed.

## 1. Emitter table (4 warnings → candidate call sites)

The 4 console lines are all the **same** log call with the **same constant id** (`8ae27e...` stable across
restarts per triage). A constant id across restarts is exactly what Streamlit 1.60 produces for a
key-less `@st.fragment`: `fragment_id = calc_hash(module + qualname + delta_path)` (`fragment.py:486-488`;
`@st.fragment` takes only `run_every/parallel`, **no `key=` parameter exists**, so *every* fragment site
in this repo lacks an explicit key by construction). The id is position-derived, hence restart-stable.

| # | Console warning | App-side owner (static evidence) | Key? | Can emit? |
|---|-----------------|----------------------------------|------|-----------|
| 1–4 (all four, same id `8ae27e...`) | `streamlit/runtime/app_session.py:456-460`, `_LOGGER.info("The fragment with id %s does not exist anymore ...")` inside `AppSession.request_rerun` (def at `:419`), fired when `client_state.fragment_id` is absent from fragment storage (`:455`) | `src/ui/components/chat_build.py:161-162` `@st.fragment(run_every=1.5)` enclosing `def _render_build_progress_fragment(sid)` (`:162`), called from `src/ui/components/chat.py:418` inside `render_chat_messages_area()` | No `key=` (parameter does not exist in 1.60) | **YES — only site with an auto-rerun timer** |
| — | same log call | `src/ui/components/viewer.py:250-251` `@st.fragment()` enclosing `def render_pdf_area()` (`:251`) | No `key=` (n/a) | NO — no `run_every`, never schedules timer reruns; locked by `tests/unit/test_viewer_fragment_no_poll.py:28` |
| — | n/a (not a fragment; `st.iframe` never registers a fragment id) | `src/ui/ui.py:73` (`inject_header_height_script`) and `src/ui/ui.py:132` (`inject_custom_css`), both bare `st.iframe(js, height="content")` | n/a | NO — cannot produce this message |

Stack excerpt (installed package, exact log line — the only stack/log anchor available):
`C:\Users\darkzard05\AppData\Roaming\Python\Python310\site-packages\streamlit\runtime\app_session.py:455-461`
(`if fragment_id and not self._fragment_storage.contains(fragment_id):` → `_LOGGER.info(...)` → `return`).
No app-repo stack was captured (see §4 blocker).

## 2. Why 4x per full-app rerun

- The sole auto-rerun timer in `src/` is `_render_build_progress_fragment` (`run_every=1.5`). While a build runs,
  `chat.py:417-418` takes the fragment branch, the id is registered, and the browser schedules 1.5 s auto-reruns.
- When the build finishes, the next **full-app rerun** takes the `else` branch (`chat.py:419-420`,
  static `_render_build_progress_block`), so the fragment id is **not re-registered** and `clear()` evicts it —
  exactly the `request_rerun:455` early-return path quoted above.
- Timer requests already queued/in flight from the browser still arrive with the stale id; each one logs the
  INFO line once and is dropped (`:461 return`). **4 lines = 4 stale timer firings for one fragment id** landing
  during/after the evicting full-app rerun (backlog accumulated while the full rerun was in flight), not
  4 distinct fragments. Consistent with: single constant id, single `run_every` site, single call site
  (`chat.py:418`, called once per `render_main_content`, `ui.py:175`).
- Corroborating repo-wide sweep: `run_every` occurs exactly once under `src/` (`chat_build.py:161`); all other
  hits are docs/comments/tests/stale AGENTS.md notes. `render_pdf_area` (`viewer.py:250`) and both `st.iframe`
  sites (`ui.py:73,132`) cannot queue fragment reruns.

## 3. Candidate-site key audit (MUST-DO a)

- `chat_build.py:161` `@st.fragment(run_every=1.5)` — no `key=` (no such parameter in 1.60).
- `viewer.py:250` `@st.fragment()` — no `key=`, no `run_every` (polling removed, fix-001).
- `ui.py:73`, `ui.py:132` `st.iframe(...)` — not fragments; `key=` n/a.
- Verdict support: only the `chat_build.py:161` fragment owns a live timer, hence only it can go stale-then-fire.

## 4. Live attempt (MUST-DO b) and blocker (MUST-DO c)

- `python -c "import streamlit"` → `1.60.0` importable. Repo grep for `does not exist anymore` → **no matches**
  (confirms Streamlit-internal origin, matching triage).
- ONE headless attempt: throwaway `reports/_tmp_measure_fragment.py` (since removed) installing a
  `warnings.showwarning`+traceback hook **and** a logging handler on `streamlit.runtime.app_session`, then
  `AppTest.from_string` minimal `@st.fragment(run_every=1)` app run twice (initial + full rerun).
  Console output grepped for the warning string:
  `WARNINGS_CAPTURED: []`, `APPSESSION_LOGS: []`, `TARGET_FOUND: False`; real module import OK
  (`ui.components.chat_build._render_build_progress_fragment` present).
- Blocker: the message is emitted via `_LOGGER.info` (INFO log, **not** `warnings.warn`), so a `showwarning`
  hook can never catch it; and the stale-id race requires browser auto-rerun timers + `AppSession.request_rerun`,
  neither of which `AppTest` exercises headlessly. A live stack needs a real browser session driving the
  1.5 s timer across a build-completion full rerun — out of scope for this attribution task.
- Hence **STATIC-PARTIAL**, not a stack-proven emitter claim.

## 5. Rule compliance

- No emitter claimed from a stack: the §1 table marks the single-timer owner as static evidence only.
- `src/` untouched (read-only), no keys added, nothing committed, files only in `reports/` (+ OS temp
  `tmpctzlsgtk` auto-dir from the probe, outside repo).

## 6. Verdict

`EMITTER: src/ui/components/chat_build.py:161` (STATIC-PARTIAL — `_render_build_progress_fragment`, 1.5 s timer fragment; 4x = 4 stale auto-rerun firings for its single evicted id; Streamlit-internal log site `streamlit/runtime/app_session.py:456`)
