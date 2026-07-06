# 프로젝트 안정성 복구 및 강화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 유실된 스트리밍 로직을 복구하고 상태 접근 및 동기화 문제를 해결하여 전체 단위 테스트를 통과시킵니다.

**Architecture:** 방어적 상태 접근 패턴 적용, 전역 Threading Lock 기반 리소스 관리, 스트리밍 핸들러 복원.

**Tech Stack:** Python 3.10, LangGraph, Streamlit, Pydantic v2.

---

### Task 1: 스트리밍 핸들러 복구 (src/api/streaming_handler.py)

**Files:**
- Modify: `src/api/streaming_handler.py`
- Test: `tests/unit/test_streaming_buffer.py`, `tests/unit/test_streaming_ttft.py`

- [ ] **Step 1: 유실된 클래스 및 메서드 복원**
    - `TokenStreamBuffer`에 `add_token`, `flush`, `reset` 메서드 구현 (TTFT 최적화 포함)
    - `StreamingResponseHandler`에 `stream_graph_events` 및 `stream_response` 구현
    - `StreamChunk` 데이터 클래스에 필요한 필드 (`thought`, `status` 등) 추가
- [ ] **Step 2: 단위 테스트 실행 및 확인**
    - Run: `pytest tests/unit/test_streaming_buffer.py tests/unit/test_streaming_ttft.py`
    - Expected: PASS

### Task 2: ResourcePool 스레드 안전성 강화 (src/core/resource_pool.py)

**Files:**
- Modify: `src/core/resource_pool.py`
- Test: `tests/unit/test_resource_pool_concurrency.py`

- [ ] **Step 1: threading.Lock 도입 및 로직 수정**
    - `_local` 및 `asyncio.Lock` 제거
    - 클래스 레벨 `_lock = threading.Lock()` 추가
    - `register`, `get`, `unregister`, `clear` 메서드를 `with self._lock:` 블록으로 보호
- [ ] **Step 2: 동시성 테스트 실행**
    - Run: `pytest tests/unit/test_resource_pool_concurrency.py`
    - Expected: PASS

### Task 3: GraphState 방어적 접근 적용 (src/core/graph_builder.py)

**Files:**
- Modify: `src/core/graph_builder.py`
- Test: `tests/unit/test_graph_flow.py`, `tests/unit/test_structured_nodes.py`, `tests/unit/test_reducer_state.py`

- [ ] **Step 1: get_state_attr 유틸리티 추가 및 적용**
    - 파일 상단에 `get_state_attr` 함수 정의
    - `preprocess`, `grade_documents`, `rewrite_query`, `generate` 등 모든 노드에서 `state.attr` 접근을 `get_state_attr(state, "attr")`로 교체
- [ ] **Step 2: 그래프 흐름 테스트 실행**
    - Run: `pytest tests/unit/test_graph_flow.py tests/unit/test_structured_nodes.py tests/unit/test_reducer_state.py`
    - Expected: PASS

### Task 4: 최종 통합 검증 및 정리

**Files:**
- Test: `tests/unit/test_timeline_sync.py`
- Modify: `checklist.md`, `context-notes.md`

- [ ] **Step 1: UI 타임라인 동기화 테스트 확인**
    - Run: `pytest tests/unit/test_timeline_sync.py`
    - Expected: PASS (ImportError 해결 확인)
- [ ] **Step 2: 전체 단위 테스트 재확인**
    - Run: `pytest tests/unit`
    - Expected: 모든 117개 테스트 PASS
- [ ] **Step 3: 문서 업데이트 및 마무리**
    - `checklist.md`에 작업 내용 기록
    - `context-notes.md`에 기술적 결정 사항 및 결과 요약
