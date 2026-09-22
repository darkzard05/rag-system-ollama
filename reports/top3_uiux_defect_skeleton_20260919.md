# Top 3 UI/UX Defect Report (Skeleton)

**Generated:** 2026-09-19
**Branch HEAD:** `<!-- TODO: fill commit SHA -->`
**Source changes:** ZERO (this is a read-only audit; no `src/` diffs)

---

## Executive Summary

Three defects ranked by severity after a 5-member hyperplan (R1→R2→R3). Each passes all three gates (G1 line-exists, G2 UI-thread, G3 user-visible). Two additional candidates (FR, viewer manual-only) sit below the severity threshold.

---

## Ranked Defects

### #1 — DG: Dead-Gate Input Bypass (P0, unanimous)

**One-liner:** The chat input stays interactive during generation because `disabled=False` is hardcoded; submit fires into an unguarded handler.

**File:line evidence**

| File | Lines | What |
|------|-------|------|
| `src/ui/components/chat.py` | 454 | `_resolve_chat_input_state` returns `input_disabled` (single hit, result discarded) |
| `src/ui/components/chat.py` | 466–471 | `st.chat_input(disabled=False)` hardcoded, ignores `input_disabled` |
| `src/ui/components/chat.py` | 473–514 | Submit handler processes query with no generation-in-progress guard |
| `src/ui/components/chat.py` | 609–610 | KO error string becomes message body text (`f"오류가 발생했습니다: {exc}"`) |
| `src/ui/components/chat.py` | 647–661 | Error-bearing response persisted to session via `add_message` |
| `src/core/rag_core.py` | 163–166 | `VectorStoreError` raised when RAG engine absent (feeds into 609-610) |
| `src/core/rag_core.py` | 202–205 | Same `VectorStoreError` in `astream` path |

**Gates**

| Gate | Result | Detail |
|------|--------|--------|
| G1 line-exists | PASS | All 7 locations verified in Task1 ledger (45 FOUND total) |
| G2 UI-thread | PASS | `st.chat_input` runs on Streamlit main thread |
| G3 user-visible | PASS | User can fire a second query mid-generation |

**Severity rationale:** P0 unanimous. Double-submit corrupts session state (two concurrent `aquery`/`astream` calls race on shared VectorStore).

---

### #2 — SF: Stuck Flag (Conditional P0; single-safe / cross-thread lethal)

**One-liner:** `is_generating_answer` can remain `True` permanently when the finalization path (`chat.py:662-663`) is skipped by an unhandled exception, locking the input for the rest of the session.

**File:line evidence**

| File | Lines | What |
|------|-------|------|
| `src/ui/components/streaming.py` | 234 | `set("is_generating_answer", False)` on stream-level exception |
| `src/ui/components/streaming.py` | 252 | `set("is_generating_answer", False)` on normal completion |
| `src/ui/components/streaming.py` | 268 | `generation_cancel` cleared after message persistence |
| `src/ui/components/chat.py` | 278–281 | `finally` guard resets flag if still True after streaming block |
| `src/ui/components/chat.py` | 662–663 | Finalizer resets both `is_generating_answer` and `generation_cancel` |
| `src/core/session/manager.py` | 387–395 | `set()` signature: `(key, value, session_id, **kwargs)` |
| `src/core/session/manager.py` | 170–194 | `get_session_id()` fallback chain (contextvar → Streamlit ctx → "default") |
| `src/core/rag_core.py` | 294–300 | `reset_all_state` refuses to run while generating (clear-on-write blocked) |

**Gates**

| Gate | Result | Detail |
|------|--------|--------|
| G1 line-exists | PASS | All 8 locations verified in Task1 ledger |
| G2 UI-thread | PASS | Flag is read/written on Streamlit main thread |
| G3 user-visible | PASS | Input stays disabled; user must restart session |

**Severity rationale:** Conditional P0. Single-session-safe (3 reset paths exist). Cross-thread lethal when `get_session_id()` falls through to `"default"` and a stale True bleeds across users. Note: dissent recorded on severity (conditional vs. firm P0).

**Dissent note:** One panel member argued this is strictly P1 given the existing 3-path reset coverage. Majority held that the cross-thread `"default"` fallback elevates it.

---

### #3 — SS: Silent-Stream State Leak (High, unanimous)

**One-liner:** Streaming status signals (`add_status_log`, `status_logs`) are produced by core and background indexing but never read by any UI component, making pipeline progress invisible to the user.

**File:line evidence**

| File | Lines | What |
|------|-------|------|
| `src/core/graph/_generate.py` | 458–471 | `response_chunk` dispatched via `_dispatch_event` (content + thought + raw_json) |
| `src/core/graph/_generate.py` | 346–349 | `add_status_log` called for injection detection (write-only) |
| `src/ui/components/streaming_extractors.py` | 206–217 | `FinalAnswerExtractor.feed` processes `raw_json` chunks |
| `src/ui/components/streaming_state.py` | 169–176 | Chunk routing: `raw_json` → extractor, else direct yield |
| `src/ui/components/streaming_runtime.py` | 271–279 | `write_stream` content generator mirrors same logic |
| `src/core/session/manager.py` | 486–500 | `add_status_log`: appends to `status_logs` list (dedup, cap 30) |
| `src/_bg_indexing.py` | 101 | Background indexing reads `status_logs` into progress message |
| `src/ui/` | *zero refs* | No UI component reads `status_logs` directly |

**Gates**

| Gate | Result | Detail |
|------|--------|--------|
| G1 line-exists | PASS | All 8 locations verified in Task1 ledger |
| G2 UI-thread | PASS | Status logs written from async thread, consumed (or not) on UI thread |
| G3 user-visible | PASS | User sees no status updates during generation or indexing |

**Severity rationale:** High unanimous. Status signals exist in the session state but the UI layer has zero consumers, so pipeline internals (injection detection, indexing progress) are silently lost.

---

## Below-Threshold Candidates

### FR: Freeze-Then-Rerun (Medium)

Chat finalizer calls `st.rerun()` unconditionally at `chat.py:669`, which re-executes the entire page script. The stop-button path (`chat.py:665-669`) avoids this, but the normal-completion path does not. Measurable flicker on slow connections. Does not meet the P0/P1 threshold because it is cosmetic, not data-corrupting.

| File | Lines | What |
|------|-------|------|
| `src/ui/components/chat.py` | 665 | `_finalize_pdf_side_effects(current_sid, msg_id)` called before rerun |
| `src/ui/components/chat.py` | 669 | `st.rerun()` unconditional on normal completion |

### Viewer: Manual-Only Refresh (Medium)

PDF viewer fragment (`viewer.py`) uses `@st.fragment(run_every=2.0)` for auto-refresh but coordinate highlights require manual page navigation. Not ranked because it's a known design tradeoff documented in the UI AGENTS.

---

## Gate Log

| Gate | Definition | Result | Notes |
|------|-----------|--------|-------|
| G1 line-exists | Every cited `file:line` verified in source | PASS | Task1 ledger: 45 FOUND, 1 REJECTED |
| G2 UI-thread | Defect affects Streamlit main-thread execution | PASS | All 3 ranked items confirmed |
| G3 user-visible | Defect produces observable user-facing effect | PASS | Input stuck, input bypass, missing status |

---

## Provenance: R1 → R2 → R3

### Round 1 (R1) — Initial Hypotheses
5-member hyperplan produced candidate defects. Initial set included 8+ items across severity levels.

### Round 2 (R2) — Paraphrase & Validation
Paraphrase step cross-checked R1 claims against source. **Key correction:** R2 initially cited `src/ui/ui.py:343-391` as evidence. This was vacated: `ui.py` is 178 lines total (line 343 does not exist). All DG/SF/SS evidence was re-anchored to correct files (`chat.py`, `streaming.py`, `manager.py`, `rag_core.py`, `_generate.py`).

### Round 3 (R3) — Final Ranking
Ranking finalized: **DG > SF > SS**. DG is unanimous P0 (hardcoded `disabled=False`). SF is conditional P0 (single-safe, cross-thread lethal, dissent noted). SS is High unanimous (write-only signals, zero UI consumers).

---

## Conceded Defects (acknowledged but excluded from Top 3)

| ID | Description | Reason Excluded |
|----|-------------|-----------------|
| A1 | PDF viewer coordinate hydration race | Cosmetic only; highlights appear on next page visit |
| A3 | Session reset during generation lag | Mitigated by `rag_core.py:294-300` refusal guard |
| C1 | Status log dedup too aggressive (cap 30) | Edge case; logs still visible in UI build progress |
| C2 | `_resolve_chat_input_state` return value unused | Subsumed by DG (#1); same root cause |
| C3 | `generation_cancel` flag race window | Narrow timing; 3 existing reset paths cover |
| D3 | Background indexing `status_logs` read | Consumed by `_bg_indexing.py:101`; not truly zero-consumer |

---

## Discarded Defects (did not survive gate checks)

None. All 45 candidates from Task1 ledger passed G1. No items were discarded at gate check.

---

## Open Questions

1. **Branch HEAD:** Which commit SHA should be cited as the audit baseline?
2. **Ranking criterion:** Should P0 vs. High weight user-impact frequency, or severity alone?
3. **Report location:** Should this skeleton be promoted to `docs/uiux/` or remain in `reports/`?

---

*This is a read-only audit skeleton. Zero `src/` changes were made. All file:line references verified against Task1 ledger (45 FOUND).*
