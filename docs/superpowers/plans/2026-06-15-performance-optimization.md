# Phase 2: Pipeline Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 추론 병목 해제 및 인용 처리 알고리즘 고속화를 통한 전체 RAG 성능 향상.

**Architecture:** 
1. `MAX_CONCURRENT_INFERENCE` 상향을 통한 병렬 추론 허용.
2. 인용 매칭을 위해 문서 리스트를 해시 맵으로 사전 인덱싱하여 검색 속도 최적화.

**Tech Stack:** Python, LangChain, Ollama, concurrent.futures.

---

### Task 1: 추론 병목 해제 설정 및 검증

**Files:**
- Modify: `src/common/config.py`
- Test: `tests/performance/test_inference_concurrency.py`

- [ ] **Step 1: 병렬 추론을 검증하는 테스트 작성**

```python
# tests/performance/test_inference_concurrency.py
import asyncio
import time
from src.core.model_loader import ModelManager

async def test_parallel_inference_speed():
    # 시뮬레이션: 4개의 동시 추론 요청이 직렬(4초)보다 빠르게(4초 미만) 끝나는지 확인
    # 실제 모델 호출 대신 세마포어 점유 시간 측정
    pass # 구현 상세는 Task 실행 시 작성
```

- [ ] **Step 2: `src/common/config.py`에서 `MAX_CONCURRENT_INFERENCE` 수정**

```python
# src/common/config.py
MAX_CONCURRENT_INFERENCE: int = _get_env(
    "MAX_CONCURRENT_INFERENCE", _models_config.get("max_concurrent_inference", 4), int
)
```

- [ ] **Step 3: 테스트 실행 및 성능 향상 확인**

Run: `pytest tests/performance/test_inference_concurrency.py -v`
Expected: PASS (4개 병렬 요청이 기존보다 빠르게 처리됨)

- [ ] **Step 4: 커밋**

```bash
git add src/common/config.py tests/performance/test_inference_concurrency.py
git commit -m "perf: increase MAX_CONCURRENT_INFERENCE to 4 for better throughput"
```

---

### Task 2: 고속 인용 처리 알고리즘 개편

**Files:**
- Modify: `src/common/utils.py`
- Test: `tests/unit/test_citation_performance.py`

- [ ] **Step 1: 인용 처리 성능 측정용 벤치마크 테스트 작성**

```python
# tests/unit/test_citation_performance.py
import time
from src.common.utils import apply_tooltips_to_response

def test_citation_matching_performance():
    # 1000개의 문서 청크와 10개의 인용이 포함된 답변 준비
    # 처리 시간이 0.1초 이내인지 검증
```

- [ ] **Step 2: `apply_tooltips_to_response` 리팩토링 (해시 맵 도입)**

```python
# src/common/utils.py
def apply_tooltips_to_response(response_text, documents, ...):
    # 1. 문서 인덱싱 (Hash Map 생성)
    doc_map = {}
    for doc in documents:
        meta = ...
        key = (meta.get("page"), meta.get("current_section"))
        doc_map[key] = doc
        
    # 2. 인용 매칭 (O(1) 검색)
    def replace_citation(match):
        # doc_map.get((target_page, target_section))
        ...
```

- [ ] **Step 3: 테스트 실행 및 속도 개선 확인**

Run: `pytest tests/unit/test_citation_performance.py -v`
Expected: PASS (기존 대비 수십 배 속도 향상 확인)

- [ ] **Step 4: 커밋**

```bash
git add src/common/utils.py tests/unit/test_citation_performance.py
git commit -m "perf: optimize citation matching complexity from O(N*M) to O(N)"
```

---

### Task 3: 최종 성능 벤치마크 보고

- [ ] **Step 1: 전체 시스템 벤치마크 실행**

Run: `python scripts/test_full_pipeline.py` (또는 적절한 스크립트)

- [ ] **Step 2: 결과 분석 및 사용자 보고**
  - TTFT 변화
  - 전체 응답 시간 변화
  - 동시 사용자 처리 능력 보고
