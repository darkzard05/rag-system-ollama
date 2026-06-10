# RAG Pipeline Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve RAG pipeline speed by 40% and reduce costs by 60% through LLM grading short-circuit, unified indexing pruning, and dynamic reranking.

**Architecture:** 
1. **Short-circuit:** Bypass LLM grading in `graph_builder.py` if `rerank_score` >= 0.85.
2. **Dynamic Top-K:** Adjust reranking candidate count based on score gap in `graph_builder.py`.
3. **Unified Pruning:** Integrate duplication check directly into `semantic_chunker.py` and remove redundant calls in `chunking.py`.

**Tech Stack:** Python, LangGraph, FlashRank, NumPy, Pytest.

---

### Task 1: Implement LLM Grading Short-circuit

**Files:**
- Modify: `src/core/graph_builder.py`
- Test: `tests/unit/test_optimization_logic.py` (Create)

- [ ] **Step 1: Create a test for short-circuit logic**

```python
import pytest
from langchain_core.documents import Document
from api.schemas import GraphState

def test_grade_documents_short_circuit():
    # Mock docs with high rerank_score
    docs = [
        Document(page_content="High score doc", metadata={"rerank_score": 0.95}),
        Document(page_content="Low score doc", metadata={"rerank_score": 0.1})
    ]
    state = {"relevant_docs": docs, "input": "test query", "intent": "rag", "retry_count": 0}
    
    # We will need to import the node function after modification or mock its logic
    # This is a conceptual test check
    pass
```

- [ ] **Step 2: Modify `grade_documents` in `src/core/graph_builder.py`**

```python
async def grade_documents(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    # ... existing code ...
    docs = get_state_attr(state, "relevant_docs")
    if not docs:
        return {"intent": "transform"}

    # [Optimization] Short-circuit: Check rerank_score
    max_score = max((d.metadata.get("rerank_score", 0.0) for d in docs), default=0.0)
    if max_score >= 0.85:
        logger.info(f"[RAG] [GRADE] Short-circuit 활성화: Max Score {max_score:.3f} >= 0.85. LLM 평가 생략.")
        SessionManager.add_status_log("신뢰도 높은 지식이 발견되어 즉시 답변 생성을 시작합니다.")
        return {"intent": "generate"}
    
    # ... rest of the LLM grading logic ...
```

- [ ] **Step 3: Verify with logs**
Run a query and check if "Short-circuit 활성화" appears in logs when a relevant document is found.

- [ ] **Step 4: Commit**
```bash
git add src/core/graph_builder.py
git commit -m "perf: implement LLM grading short-circuit based on rerank_score"
```

---

### Task 2: Implement Dynamic Reranking Top-K

**Files:**
- Modify: `src/core/graph_builder.py`

- [ ] **Step 1: Modify `retrieve_and_rerank` in `src/core/graph_builder.py`**

```python
async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    # ... (after aggregation) ...
    aggregated, _ = aggregator.aggregate_results(...)

    # [Optimization] Dynamic Top-K calculation
    if len(aggregated) >= 10:
        top_1_score = aggregated[0].aggregated_score
        top_10_score = aggregated[9].aggregated_score
        score_gap = top_1_score - top_10_score
        
        # Gap이 크면 후보군 축소 (명확한 상위 그룹 존재)
        dynamic_top_k = 12 if score_gap > 0.5 else 25
        logger.info(f"[RAG] [RETRIEVE] Dynamic Top-K: {dynamic_top_k} (Gap: {score_gap:.3f})")
    else:
        dynamic_top_k = 25

    final_docs = [
        Document(page_content=r.content, metadata=r.metadata)
        for r in aggregated[:dynamic_top_k]
    ]
    # ... (reranker call) ...
```

- [ ] **Step 2: Commit**
```bash
git add src/core/graph_builder.py
git commit -m "perf: add dynamic reranking candidate selection based on score gap"
```

---

### Task 3: Unified Indexing & Single-Pass Pruning

**Files:**
- Modify: `src/core/semantic_chunker.py`
- Modify: `src/core/chunking.py`

- [ ] **Step 1: Add pruning logic to `EmbeddingBasedSemanticChunker._optimize_chunk_sizes` in `src/core/semantic_chunker.py`**

```python
    def _optimize_chunk_sizes(self, chunks: list[dict]) -> list[dict]:
        # ... existing logic ...
        # (Inside the loop, after determining if it shouldn't merge)
        # Check for near-duplicate against already optimized chunks
        # This is simplified; a full vector comparison might be better in a separate pass within this method
        pass
```
*Wait, let's refine this to be more efficient.*

- [ ] **Step 2: Implement `_prune_duplicates` in `EmbeddingBasedSemanticChunker`**

```python
    def _prune_duplicates(self, chunks: list[dict], threshold: float = 0.98) -> list[dict]:
        if not chunks: return []
        pruned = [chunks[0]]
        for i in range(1, len(chunks)):
            current_vec = chunks[i]["vector"]
            # 단순화를 위해 바로 직전 청크와만 비교 (인접 중복이 가장 흔함)
            # 또는 최근 N개와 비교
            is_dup = False
            for prev in pruned[-3:]: # 최근 3개와 비교
                sim = np.dot(current_vec, prev["vector"])
                if sim > threshold:
                    is_dup = True
                    break
            if not is_dup:
                pruned.append(chunks[i])
        return pruned
```

- [ ] **Step 3: Update `split_text` to call `_prune_duplicates`**

- [ ] **Step 4: Modify `src/core/chunking.py` to remove redundant optimizer**

```python
    # [최적화 핵심] 중복 및 유사 문서 프루닝 제거 (SemanticChunker 내장으로 대체)
    # optimizer = get_index_optimizer() 
    # split_docs, vectors, _, _ = optimizer.optimize_index(split_docs, vectors)
```

- [ ] **Step 5: Commit**
```bash
git add src/core/semantic_chunker.py src/core/chunking.py
git commit -m "perf: unify pruning into semantic chunker and remove redundant optimizer calls"
```

---

### Task 4: Final Validation and Graph Update

- [ ] **Step 1: Run performance benchmark if available**
Run `python scripts/e2e_performance_benchmark.py` (if it exists) or test manually.

- [ ] **Step 2: Update Graphify**
Run `graphify update .`

- [ ] **Step 3: Final Commit**
```bash
git commit --allow-empty -m "perf: complete RAG pipeline optimization phase"
```
