# src/main.py Iteration 1: 안정성 및 리소스 관리 강화 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/main.py`의 백그라운드 스레드 작업의 안정성을 높이고, 임시 파일 및 세션 리소스의 정리를 더 견고하게 만듭니다.

**Architecture:** 
1. 백그라운드 스레드(`_rebuild_rag_system`, `_update_qa_chain`) 내의 예외 발생 시 세션 상태에 상세 에러를 기록하고 사용자에게 알림을 강화합니다.
2. `atexit` 핸들러 외에 Streamlit 세션 종료 시점을 감지할 수 있는 추가적인 보조 장치를 검토합니다.
3. Windows 환경에서 파일 잠금으로 인한 삭제 실패를 방지하기 위해 `SessionManager.delete_session`의 재시도 로직을 보강합니다.

**Tech Stack:** Python, Streamlit, Threading, Pytest

---

### Task 1: 백그라운드 스레드 예외 처리 및 로깅 강화

**Files:**
- Modify: `src/main.py`
- Test: `tests/unit/test_main_background_tasks.py` (New)

- [ ] **Step 1: 실패하는 테스트 작성**
백그라운드 스레드 내에서 에러가 발생했을 때 `SessionManager`에 올바르게 에러 메시지가 기록되는지 확인하는 테스트를 작성합니다.

- [ ] **Step 2: 에러 전파 및 알림 로직 개선**
`_rebuild_rag_system`와 `_update_qa_chain` 내부의 `try-except` 블록에서 `SystemNotifier.error`를 더 명확하게 사용하도록 수정합니다.

### Task 2: 임시 파일 삭제 로직 견고화 (Windows 호환성)

**Files:**
- Modify: `src/core/thread_safe_session.py`
- Test: `tests/unit/test_session_cleanup.py`

- [ ] **Step 1: 파일 삭제 재시도 로직 강화**
`SessionManager.delete_session`에서 `PermissionError` 발생 시 지수 백오프(Exponential Backoff)를 적용하여 재시도하도록 수정합니다.

- [ ] **Step 2: 테스트를 통한 검증**
파일이 열려 있는 상태를 시뮬레이션하고 삭제 로직이 성공하는지 확인합니다.

### Task 3: 메인 루프 안정성 강화

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: 중복 스레드 생성 방지**
동일한 세션에서 이미 실행 중인 작업이 있는지 확인하는 플래그 체크를 더 엄격하게 수행합니다.
