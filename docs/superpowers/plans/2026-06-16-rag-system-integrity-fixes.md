# RAG System Integrity & Performance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical RAG system defects including session data loss, metadata corruption, inefficient model reloading, and fragile UI highlighting.

**Architecture:** Strengthen session persistence using Streamlit-aware fallback logic, enforce metadata inheritance during document splitting, implement loop-agnostic model re-binding, and add fuzzy matching for PDF annotations.

**Tech Stack:** Python, LangChain, LangGraph, Streamlit, PyMuPDF (fitz), Ollama.

---

### Task 1: Session & file_hash Persistence

**Files:**
- Modify: `src/core/session/manager.py`
- Modify: `src/core/rag_core.py`
- Test: `tests/unit/test_session_persistence.py`

- [ ] **Step 1: Write failing test for session ID recovery**
```python
import asyncio
from core.session import SessionManager

async def test_session_id_recovery_in_async():
    SessionManager.set_session_id("test-session")
    SessionManager.set("key", "value")
    
    async def background_task():
        # In a raw async task, ContextVar might be lost
        return SessionManager.get("key")
        
    result = await background_task()
    assert result == "value"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_session_persistence.py`
Expected: FAIL (returns None or default)

- [ ] **Step 3: Implement robust session ID recovery**
In `src/core/session/manager.py`, update `get_session_id` to prioritize `st.session_state` if available, and `get` to use an explicit `session_id` if provided.
In `src/core/rag_core.py`, ensure all calls to `SessionManager.get/set` pass `self.session_id`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/unit/test_session_persistence.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/core/session/manager.py src/core/rag_core.py tests/unit/test_session_persistence.py
git commit -m "fix: enhance session persistence and file_hash recovery"
```

---

### Task 2: Metadata Inheritance in Chunking

**Files:**
- Modify: `src/core/chunking.py`
- Test: `tests/unit/test_metadata_inheritance.py`

- [ ] **Step 1: Write failing test for metadata loss**
```python
from langchain_core.documents import Document
from core.chunking import split_documents

async def test_metadata_inheritance():
    docs = [Document(page_content="test", metadata={"file_path": "path/to/test.pdf", "file_hash": "abc"})]
    # Simulate chunking that might drop metadata
    chunks, _ = await split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata.get("file_path") == "path/to/test.pdf"
        assert chunk.metadata.get("file_hash") == "abc"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_metadata_inheritance.py`
Expected: FAIL (AssertionError: file_path is None)

- [ ] **Step 3: Implement metadata audit in chunking**
In `src/core/chunking.py`, update `_postprocess_metadata` to take the original `docs` as a reference and re-inject `file_path` and `file_hash` if missing.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/unit/test_metadata_inheritance.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/core/chunking.py tests/unit/test_metadata_inheritance.py
git commit -m "fix: ensure metadata inheritance during document splitting"
```

---

### Task 3: Async Loop Hotswap Optimization

**Files:**
- Modify: `src/core/model_loader.py`
- Test: `tests/performance/test_loop_hotswap.py`

- [ ] **Step 1: Write test to measure hotswap time**
```python
import time
from core.model_loader import ModelManager

async def test_hotswap_performance():
    # Load once
    await ModelManager.get_embedder("nomic-embed-text-v2-moe")
    
    # Simulate loop change by clearing internal loop ID (mock)
    ModelManager._last_loop_id = 0 
    
    start = time.time()
    await ModelManager.get_embedder("nomic-embed-text-v2-moe")
    duration = time.time() - start
    assert duration < 0.5  # Should be very fast
```
- [ ] **Step 2: Run test to verify it fails (slow reload)**
Run: `pytest tests/performance/test_loop_hotswap.py`
Expected: FAIL (duration > 0.5s due to full reload)

- [ ] **Step 3: Implement partial re-binding in ModelManager**
Modify `_ensure_loop_integrity` in `src/core/model_loader.py`. Instead of `cls._models.clear()`, only re-initialize the LangChain wrapper objects (which are lightweight) but keep the underlying model binaries/VRAM resources if possible.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/performance/test_loop_hotswap.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/core/model_loader.py tests/performance/test_loop_hotswap.py
git commit -m "perf: optimize model hotswap by re-binding to new async loop"
```

---

### Task 4: Fuzzy Coordinate Matching for Highlights

**Files:**
- Modify: `src/common/utils.py`
- Test: `tests/unit/test_fuzzy_highlight.py`

- [ ] **Step 1: Write failing test for strict matching**
```python
from common.utils import extract_annotations_from_docs

def test_fuzzy_highlight_matching():
    # Slightly different text from what's in 'word_coords'
    doc = {
        "page_content": "This is a test response",
        "metadata": {
            "page": 1,
            "word_coords": [(0,0,10,10,"This"), (11,0,20,10,"is"), (21,0,30,10,"a"), (31,0,50,10,"TEST"), (51,0,100,10,"response")],
            "file_path": "dummy.pdf"
        }
    }
    # Current strict matching might fail on "test" vs "TEST" if normalization is weak
    # or if a word is missing.
    annos = extract_annotations_from_docs([doc])
    assert len(annos) > 0
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_fuzzy_highlight.py`
Expected: FAIL (len(annos) == 0)

- [ ] **Step 3: Implement fuzzy matching fallback**
In `src/common/utils.py`, update `extract_annotations_from_docs`. If `filtered_coords` is empty after strict token matching, use a sliding window with a character-level similarity threshold (e.g., using `difflib`).

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/unit/test_fuzzy_highlight.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/common/utils.py tests/unit/test_fuzzy_highlight.py
git commit -m "fix: implement fuzzy matching for PDF highlight annotations"
```
