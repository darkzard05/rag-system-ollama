# Specification: Phase 1 - Reliability Stabilization

**Status:** Draft
**Date:** 2026-06-15
**Phase:** 1 of 3
**Topic:** RAG 파이프라인 안정화 (PDF 핸들 누수 방지 및 세션 동기화 개선)

## 1. 개요 (Overview)
현재 시스템에서 발견된 두 가지 치명적 결함(PDF 파일 핸들 누수, 세션 상태 경합)을 해결하여 시스템의 장기 운영 안정성을 확보합니다.

## 2. 목표 (Goals)
- PDF 파싱 과정에서 발생하는 예외와 관계없이 모든 파일 리소스를 즉시 해제.
- 다중 스레드 환경(UI 및 백그라운드 워커)에서 세션 데이터 정합성 보장.
- 성능 저하 없는 세션별 독립적 잠금(Locking) 메커니즘 구축.

## 3. 상세 설계 (Detailed Design)

### 3.1. PDF Resource Guardian
- **파일:** `src/core/document_processor.py`
- **변경 사항:**
  - `open_pdf_document` 컨텍스트 매니저를 `load_pdf_docs` 함수 내부에 적용.
  - 1차(지능형), 2차(테이블 제외), 3차(C-Engine) 모든 파싱 경로를 단일 `try-finally` 구조로 보호.
  - 파싱 중단이나 크래시 발생 시에도 `doc.close()` 실행 보장.

### 3.2. Thread-Safe Session Manager (Approach C)
- **파일:** `src/core/session/manager.py`, `src/core/rag_core.py`
- **변경 사항:**
  - `_session_locks: dict[str, threading.RLock]` 도입.
  - `set()`, `update()` 호출 시 해당 세션의 전용 락 획득 후 데이터 수정.
  - `sync_to_streamlit()` 메서드를 통해 렌더링 직전에만 `st.session_state`로 데이터 단방향 동기화.
  - 백그라운드 태스크에서 `st.session_state` 직접 수정을 금지하고 `SessionManager`를 경유하도록 강제.

## 4. 데이터 흐름 (Data Flow)
1. **Worker Thread:** 데이터 발생 → `SessionManager.set(key, val, sid)` (Lock 획득) → 내부 저장소 기록 → Lock 해제.
2. **UI Thread:** 리렌더링 시작 → `SessionManager.sync_to_streamlit(sid)` → `st.session_state` 업데이트 → 화면 출력.

## 5. 검증 전략 (Verification Strategy)

### 5.1. 리소스 누수 테스트 (`tests/stability/test_pdf_leak.py`)
- **방법:** 손상된 PDF 100개를 연속 로드하여 에러를 유도한 후 `psutil.Process().num_fds()` 변화 측정.
- **성공 기준:** 테스트 종료 후 FD(File Descriptor) 증가량이 0이어야 함.

### 5.2. 세션 동시성 테스트 (`tests/stability/test_session_concurrency.py`)
- **방법:** `ThreadPoolExecutor`를 사용하여 10개 스레드에서 동일 세션의 `set/get`을 1000회 반복 수행.
- **성공 기준:** 데이터 누락이나 `RuntimeError` 없이 100% 성공해야 함.

## 6. 예외 처리 (Error Handling)
- 타임아웃 발생 시 스트리밍을 중단하고 해당 세션의 락 상태를 강제 초기화.
- 리소스 로딩 최종 실패 시 로그를 남기고 세션 상태에 사용자 알림 메시지 기록.
