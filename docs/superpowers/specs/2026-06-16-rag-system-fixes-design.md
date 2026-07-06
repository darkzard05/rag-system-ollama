# Design: RAG System Integrity & Performance Fixes

## 1. Overview
This design addresses four critical issues identified in the RAG system logs: session data volatility, metadata loss during chunking, inefficient async loop handling, and fragile UI highlighting.

## 2. Problem Statements
1. **VectorStoreError**: `file_hash` is lost during async execution, causing query failures.
2. **Metadata Loss**: `file_path` is missing in processed chunks, breaking citations.
3. **Async Loop Latency**: Frequent loop changes in Streamlit cause expensive model reloads.
4. **Highlight Failure**: Strict token matching prevents citations from being highlighted in the PDF viewer.

## 3. Proposed Solutions

### 3.1. Session Persistence (Issue 1)
- **Mechanism**: Implement a `SessionContext` wrapper that ensures the correct `session_id` is propagated to background tasks.
- **Robustness**: Modify `SessionManager` to fallback to `st.session_state` more aggressively when `ContextVar` is empty in a UI-originating thread.

### 3.2. Metadata inheritance (Issue 2)
- **Mechanism**: Update `src/core/chunking.py` to perform a "metadata audit" after splitting.
- **Logic**:
  ```python
  for chunk in chunks:
      if not chunk.metadata.get("file_path"):
          chunk.metadata["file_path"] = source_docs[0].metadata["file_path"]
  ```

### 3.3. Loop Re-binding Strategy (Issue 3 - Approach B)
- **Mechanism**: Decouple "Heavy Model Data" (VRAM) from "Execution Primitives" (Async Clients/Locks).
- **Optimization**: When `ModelManager` detects a loop change, it will only re-instantiate the LangChain model wrapper's internal client/session, keeping the underlying model loaded in memory/VRAM.

### 3.4. Fuzzy Highlighting (Issue 4)
- **Mechanism**: Implement a fallback matching algorithm in `extract_annotations_from_docs`.
- **Logic**: Use `difflib.SequenceMatcher` or a sliding window character-level match if token-level alignment fails.

## 4. Components to be Modified
- `src/core/session/manager.py`: Session ID resolution and fallback logic.
- `src/core/rag_core.py`: Explicit session passing and configuration validation.
- `src/core/chunking.py`: Metadata validation post-split.
- `src/core/model_loader.py`: Partial reload (re-binding) logic.
- `src/common/utils.py`: Robust annotation extraction.

## 5. Success Criteria
- No `VectorStoreError` during consecutive queries.
- `file_path` exists in all chunk logs.
- Model "hotswap" time reduced from ~5s to <100ms.
- Highlighting works even when response text has minor formatting differences from PDF text.
