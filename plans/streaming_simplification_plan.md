# Stream Pipeline Simplification Plan

> **Status (2026-10-11): SUPERSEDED — premises stale, do not execute as-is.**
> - `RAGOrchestrator` no longer exists in `src` (0 references).
> - Current pipeline: `src/core/rag_core.py:astream` → `src/api/stream_pipeline.py:stream_graph_events`
>   (with `TokenStreamBuffer`) → `src/ui/components/streaming_core.py` consumption.
>   There is no `astream`/`astream_events` merge target and no `StreamingResponseHandler`
>   removal target as described below.
> - A new plan must be written against the current pipeline after H2/H4 quality work.
>   No TTFT benchmark script exists yet.
>
> Original content preserved below for archaeology.

## 1. Goal
Simplify the RAG streaming pipeline to reduce latency (TTFT), improve reliability, and remove over-engineered layers (Queue, redundant buffers).

## 2. Strategy
- **Direct Streaming:** Remove the producer-consumer pattern in `RAGOrchestrator`.
- **Minimal Buffering:** Disable or minimize `TokenStreamBuffer` in `StreamingResponseHandler`.
- **Deterministic Routing:** Use LangGraph event metadata for clean separation of status vs. content.

## 3. Checklist
- [ ] Research: Trace the full path from `src/main.py` to `src/core/rag_core.py` to `src/api/streaming_handler.py`.
- [ ] Create reproduction/benchmark script to measure current TTFT.
- [ ] Refactor `src/core/rag_core.py`:
    - Merge `astream` and `astream_events`.
    - Remove `asyncio.Queue`.
- [ ] Refactor `src/api/streaming_handler.py`:
    - Simplify `stream_graph_events`.
    - Disable `TokenStreamBuffer` logic for local Ollama.
- [ ] Update UI integration in `src/main.py` if necessary.
- [ ] Verification: Run tests and compare TTFT results.

## 4. Risks
- **UI Flickering:** If buffering is completely removed, rapid UI updates might cause flickering in Streamlit.
- **Event Loss:** Ensure all custom events (status updates) are still captured correctly.
