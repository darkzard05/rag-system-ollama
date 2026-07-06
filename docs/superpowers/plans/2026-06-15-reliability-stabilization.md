# Phase 1: Reliability Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RAG 파이프라인의 PDF 리소스 누수 방지 및 세션 상태 동기화의 스레드 안전성 확보.

**Architecture:** 
1. `contextlib.contextmanager`를 사용하여 PDF 파일 핸들의 자동 정리를 보장하는 Resource Guardian 패턴 적용.
2. 세션 ID별 독립적인 `threading.RLock`을 사용하는 Per-Session Lock 전략(Approach C)을 통해 데이터 경합 방지.
3. UI(Streamlit)와 백그라운드 워커 간의 상태 동기화를 단방향 Sync 메서드로 분리.

**Tech Stack:** Python, threading, contextlib, psutil, pytest, pymupdf4llm.

---

### Task 1: PDF Resource Guardian 구현 및 리팩토링

**Files:**
- Modify: `src/core/document_processor.py`
- Test: `tests/stability/test_pdf_leak.py`

- [ ] **Step 1: 파일 핸들 누수 검증을 위한 실패하는 테스트 작성**

```python
# tests/stability/test_pdf_leak.py
import os
import psutil
import pytest
from core.document_processor import load_pdf_docs

def test_file_descriptor_leak_on_failure():
    process = psutil.Process(os.getpid())
    initial_fds = process.num_fds()
    
    # 존재하지 않거나 손상된 파일로 에러 유도 (10회 반복)
    for _ in range(10):
        try:
            load_pdf_docs("non_existent.pdf", "test.pdf")
        except Exception:
            pass
            
    final_fds = process.num_fds()
    # 누수가 있다면 FD가 증가함
    assert final_fds <= initial_fds + 2 
```

- [ ] **Step 2: 테스트 실행 및 실패 확인**

Run: `pytest tests/stability/test_pdf_leak.py -v`
Expected: FAIL (현재 구조에서 예외 발생 시 close()가 누락되는 경로가 있음)

- [ ] **Step 3: `open_pdf_document` 컨텍스트 매니저 도입 및 `load_pdf_docs` 리팩토링**

```python
# src/core/document_processor.py 상단에 추가
import contextlib
import fitz

@contextlib.contextmanager
def open_pdf_document(file_path: str):
    doc = None
    try:
        doc = fitz.open(file_path)
        yield doc
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass

# load_pdf_docs 함수 내부 수정 (생략된 기존 로직 포함)
def load_pdf_docs(file_path: str, file_name: str, ...):
    with open_pdf_document(file_path) as doc:
        # 1, 2, 3차 시도 로직을 이 블록 안으로 이동
        # 모든 시도에서 'doc' 객체를 사용하며, 블록 종료 시 자동 close()
        ...
```

- [ ] **Step 4: 테스트 실행 및 통과 확인**

Run: `pytest tests/stability/test_pdf_leak.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/core/document_processor.py tests/stability/test_pdf_leak.py
git commit -m "fix: ensure PDF handles are closed using context manager"
```

---

### Task 2: Thread-Safe Session Manager 구현 (Per-Session Lock)

**Files:**
- Modify: `src/core/session/manager.py`
- Test: `tests/stability/test_session_concurrency.py`

- [ ] **Step 1: 세션 경합 조건을 검증하는 실패하는 테스트 작성**

```python
# tests/stability/test_session_concurrency.py
import threading
from concurrent.futures import ThreadPoolExecutor
from core.session import SessionManager

def test_session_state_concurrency():
    SessionManager.init_session("test_concurrent_sid")
    
    def update_task(i):
        # 동일 세션에 대해 동시 업데이트 시도
        SessionManager.set("counter", i, session_id="test_concurrent_sid")
        return SessionManager.get("counter", session_id="test_concurrent_sid")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(update_task, range(100)))
        
    assert len(results) == 100
```

- [ ] **Step 2: 테스트 실행 및 실패 확인 (혹은 불안정성 확인)**

Run: `pytest tests/stability/test_session_concurrency.py -v`
Expected: FAIL (혹은 Streamlit 컨텍스트 에러/경합 발생)

- [ ] **Step 3: `SessionManager`에 세션별 RLock 도입 및 `set/update` 수정**

```python
# src/core/session/manager.py
import threading

class SessionManager:
    _session_locks: dict[str, threading.RLock] = {}
    _global_lock = threading.Lock()

    @classmethod
    def _get_lock(cls, session_id: str) -> threading.RLock:
        with cls._global_lock:
            if session_id not in cls._session_locks:
                cls._session_locks[session_id] = threading.RLock()
            return cls._session_locks[session_id]

    @classmethod
    def set(cls, key: str, value: Any, session_id: str | None = None):
        sid = session_id or cls.get_session_id()
        lock = cls._get_lock(sid)
        with lock:
            # 내부 저장소 업데이트 로직
            state = cls._get_state(sid)
            state[key] = value
            # st.session_state 직접 수정은 피함 (sync_to_streamlit에서 처리)
```

- [ ] **Step 4: 테스트 실행 및 통과 확인**

Run: `pytest tests/stability/test_session_concurrency.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/core/session/manager.py tests/stability/test_session_concurrency.py
git commit -m "feat: implement per-session RLocks for thread-safe state management"
```

---

### Task 3: Streamlit UI 단방향 동기화 및 최종 통합

**Files:**
- Modify: `src/core/session/manager.py`
- Modify: `src/main.py`

- [ ] **Step 1: `sync_to_streamlit` 메서드 구현**

```python
# src/core/session/manager.py
    @classmethod
    def sync_to_streamlit(cls, session_id: str | None = None):
        """UI 스레드에서 호출되어 내부 상태를 st.session_state로 복사"""
        if not cls._is_streamlit_running(): return
        
        sid = session_id or cls.get_session_id()
        state = cls._get_state(sid)
        lock = cls._get_lock(sid)
        
        with lock:
            for k, v in state.items():
                if k in ["pdf_processed", "is_generating_answer", "status_logs", "messages"]:
                    import streamlit as st
                    st.session_state[k] = v
```

- [ ] **Step 2: `main.py`의 렌더링 직전 동기화 호출 추가**

```python
# src/main.py
def main():
    # ... 세션 초기화 ...
    SessionManager.sync_to_streamlit()
    # ... UI 렌더링 로직 ...
```

- [ ] **Step 3: 통합 테스트 실행 (기존 테스트 포함)**

Run: `pytest tests/unit/ tests/stability/ -v`
Expected: ALL PASS

- [ ] **Step 4: 커밋**

```bash
git add src/core/session/manager.py src/main.py
git commit -m "feat: add one-way state synchronization to Streamlit UI"
```
