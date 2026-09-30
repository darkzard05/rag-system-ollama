# Repetition measurement — raw LLM completion vs final rendered body (Wave-0 Task 2)

- Mode: **STATIC-PARTIAL** (no live same-turn capture claimed)
- Timestamp: 20260926_152916 · Symptom: answer body repeats same claim ~4x
- Scope: read-only under `src/`; no fixes, no commits, no files outside `reports/`
- Predecessor: `reports/measure_citation_20260926_142832.md` (+`.json`) — shared-context evidence

`VERDICT: raw-side`

## 1. What was read (generation boundary, verbatim)

- `src/core/graph/_generate.py`
  - `format_context` (L64-81): loops `for d in docs`, appends every doc's full
    `page_content` with a `[doc:stable_id] [section] [page] [score]` header.
    No dedup keyword in the function body (`dedup|unique|seen` → no match).
  - `_apply_ctx_guard` (L115-152): trims only when estimated tokens exceed
    `(num_ctx - num_predict) * 0.85`, dropping lowest-`rerank_score` docs, min 2
    kept. No content-dedup keyword; re-formats via `format_context`.
  - `generate` (L438-474): accumulates `full_response += content_chunk` verbatim
    per streamed chunk; dispatches each chunk as a `response_chunk` event.
    Final `result["response"]` = `parsed_answer.final_answer` (or raw fallback).
- `src/ui/components/streaming_core.py`
  - `_content_generator` (L764-847): `accumulated += delta` (raw_json branch,
    L836) / `accumulated += chunk.content` (plain branch, L843); yields each
    delta unchanged. `_FinalAnswerExtractor.feed` returns incremental
    `final_answer` slices only — no expansion, no repetition logic.
  - `stream_chunks` + aux state (L867-876): citations/thought/metrics ride aux
    state; body travels only via `content` deltas.
- `src/ui/components/chat.py` `_render_streaming_with_write_stream` (L626-695):
  `response = st.write_stream(_content_generator(...))`, then persists
  `str(response)` (else `aux_state["content"]`) verbatim via
  `SessionManager.add_message` (`msg_type="general"`, `processed_content=None`).
- `src/ui/components/streaming.py` `consume_stream_into_message` (L286-294):
  same verbatim accumulation (`accumulated += delta / chunk.content`).
- `src/core/session/manager.py`
  - `add_message` (L470-511): stores `content` as-is (`"content": content`);
    same-`msg_id` updates overwrite, otherwise append. No text transform.
  - `sync_to_streamlit` (L269-336) / `UIBridge.sync_session` (`src/ui/bridge.py`
    L15-41): dirty-key (`messages`, …) snapshot mirror only; no re-emission,
    no duplication of body text.

## 2. Live-backend check (probe, not repro)

- `ollama list`: OK — `qwen3:4b-instruct-2507-q4_K_M`, `nomic-embed-text-v2-moe`
  present. Streamlit app: **not running** (only the probing process matched).
  458 PDFs under `data/temp`.
- Blocker (same as Task 1): the incident turn's **source PDF / file_hash /
  session transcript is unrecorded** — no mapping of the repeating-answer query
  to a file exists. A fresh run now would index an arbitrary PDF and produce
  different chunks — **not** the same-turn repro — so no live capture was
  attempted. No live raw completion is claimed.

## 3. Deciding measurement (mocked harness, real code paths)

Harness: 5 synthetic docs, each containing the claim sentence once
(claim = `The reserve ratio is 12 percent.`), pages [3,10,11,14,19] mirroring
the Task-1 page set; real `format_context` + `_apply_ctx_guard` imported from
`src/core/graph/_generate`; render path simulated as a verbatim port of the
accumulation lines cited in §1 (which contain no branching transform).

- (a) Context duplication survives to the prompt:
  `format_context` output contains the claim **5x** (`CTX_HAS_5_CLAIMS: 5`,
  `CTX_LEN: 662`); `_apply_ctx_guard` removed **0** docs (`GUARD_REMOVED: 0`,
  `KEPT: 5`, `CLAIMS_AFTER_GUARD: 5`). Small prompt (≈662 chars) never trips
  the token budget, so duplicated chunks reach the LLM untouched.
- (b) Repetitive raw → repetitive rendered (no render change):
  `RAW_A` (claim 4x + tail, `RAW_COUNT: 4`) split into 3 streaming deltas and
  joined verbatim → `RENDER_COUNT: 4`, `IDENTICAL: True`, difflib
  `DIFF_LINES: 0`. Trimmed raw: `The reserve ratio is 12 percent. The reserve
  ratio is 12 percent. The reserve ratio is 12 percent. The reserve ratio is
  12 percent. See citations.` — rendered body is byte-identical.
- (c) Clean raw → clean rendered (render cannot create repetition):
  `RAW_B` (claim 1x) through the same verbatim join → `CLEAN_RENDER: 1`,
  `SAME: True`. A 4x rendered body can therefore not arise from a 1x raw body
  via this path.
- (d) Extractor is transparent: `_FinalAnswerExtractor` fed a raw-JSON blob in
  3 fragments returned exactly the embedded 4x `final_answer`
  (`EXTRACTOR_OUT_COUNT: 4`, `EXTRACTOR_EQUALS_FA: True`) — it slices, never
  multiplies.

difflib diff (raw_A vs rendered_A): empty (0 diff lines) — the two strings are
identical, so there is no diff hunk to show; that emptiness IS the measurement:
the render path contributes zero edits.

## 4. Relation to Task-1 branch D (shared duplicated-context suspect)

Consistent and mutually reinforcing. Task 1 proved the retriever hands the
renderer 5 same-section chunks on distinct pages by construction (branch D).
This task proves the generator hands the LLM all 5 chunks' full texts by
construction (no dedup in `format_context`, budget-only trim that removed 0).
The same duplicated chunks therefore explain both symptoms through one cause:
duplicated context in → LLM echoes the repeated claim ~4x in the raw
completion → verbatim render pipeline reproduces it 1:1. No second
(double-append / re-emission) mechanism is needed, and none was found in the
render path (every stage audited above is append/store/mirror verbatim).

## 5. Task-6 live rerun requirements (NOT captured — no live turn)

Record source PDF path + `file_hash` + session transcript; capture the true raw
`full_response` (generate L457) and the persisted message `content`; count the
claim sentence in each with difflib; re-apply this verdict rule (raw 4x →
raw-side stands; raw 1x + rendered 4x → overturn to render-side).

`VERDICT: raw-side`
