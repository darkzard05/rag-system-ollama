# Loop-Bound Resource Management Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the RAG system handles event loop changes gracefully, preventing crashes and hangs in multi-loop environments like Streamlit.

**Architecture:** 
1. Enhance `ModelManager` to detect event loop changes and automatically clear/recreate all loop-bound resources (locks, semaphores, clients).
2. Update `RAGSystem` to invalidate session-cached retrievers when a loop change is detected.
3. Consolidate loop-awareness logic to prevent redundant checks.

**Tech Stack:** Python, asyncio, LangGraph, LangChain

---

### Task 1: Enhance `ModelManager` with Comprehensive Loop Awareness

**Files:**
- Modify: `src/core/model_loader.py`

- [ ] **Step 1: Update `_get_from_cache` to be more aggressive**
Currently, it only pops specific keys. It should handle all loop-bound state if a change is detected.

- [ ] **Step 2: Make `_get_lock` and `_get_semaphore` loop-aware**
Add a mechanism to track the current loop for locks and semaphores.

```python
    _resource_loop_id: int = 0

    @classmethod
    def _ensure_loop_integrity(cls):
        """현재 루프가 리소스가 생성된 루프와 일치하는지 확인하고, 다르면 초기화합니다."""
        try:
            current_loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            return

        if cls._resource_loop_id != 0 and cls._resource_loop_id != current_loop_id:
            logger.info(f"[ModelManager] 글로벌 리소스 루프 변경 감지 ({cls._resource_loop_id} -> {current_loop_id}). 전역 락/세마포어를 초기화합니다.")
            cls._locks.clear()
            cls._inference_semaphore = None
            cls._async_client = None
            cls._client_loop = None
            # 인스턴스들도 루프에 묶여있을 수 있으므로 정리 (기존 _get_from_cache 로직과 상보적)
            cls._instances.clear()
            cls._instance_loops.clear()
            
        cls._resource_loop_id = current_loop_id
```

- [ ] **Step 3: Call `_ensure_loop_integrity` in all entry points**
Call it in `get_llm`, `get_embedder`, `get_async_client`, `inference_session`, etc.

### Task 2: Update `RAGSystem` to Invalidate Cached Retrievers

**Files:**
- Modify: `src/core/rag_core.py`

- [ ] **Step 1: Modify `_get_rag_engine` to clear session retrievers on loop change**
When `rag_engine` is rebuilt, also clear `active_faiss_retriever` and `active_bm25_retriever`.

```python
            if rag_engine:
                logger.info(...)
                # 루프 변경 시 세션에 캐싱된 리트리버들도 무효화
                SessionManager.delete("active_faiss_retriever", session_id=self.session_id)
                SessionManager.delete("active_bm25_retriever", session_id=self.session_id)
```

### Task 3: Verification

- [ ] **Step 1: Run reproduction script and verify it passes**
- [ ] **Step 2: Run existing unit and integration tests**
- [ ] **Step 3: Verify with a mock Streamlit-like multi-loop test**

---
Plan complete and saved to `docs/superpowers/plans/2026-05-29-loop-integrity-fix.md`.
 접근 방식 선택:
1. Subagent-Driven (추천)
2. Inline Execution
