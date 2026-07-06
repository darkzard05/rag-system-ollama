# Stream Pipeline Simplification Plan

## 1. Goal
Simplify the RAG streaming pipeline to reduce latency (TTFT), improve reliability, and remove over-engineered layers (Queue, redundant buffers).

## 2. Strategy
- **Direct Streaming:** Remove the producer-consumer pattern in `RAGSystem`.
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
