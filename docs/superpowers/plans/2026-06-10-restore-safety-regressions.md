# Restoration of Safety Checks and Integrity Logic in src/main.py

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore critical safety checks, Windows integrity logic, `nest_asyncio` support, and comprehensive documentation in `src/main.py` that were removed during Task 1.

**Architecture:** Re-integrate removed logic into the existing Streamlit entry point while maintaining the new `sidebar_auto_collapsed` state management.

**Tech Stack:** Python, Streamlit, `nest_asyncio`, `torch` (for integrity check).

---

### Task 1: Restore Imports and Global Configurations

**Files:**
- Modify: `src/main.py:1-40`

- [ ] **Step 1: Re-add `nest_asyncio` import and call `apply()`**
- [ ] **Step 2: Restore missing docstrings and comments for global setup**

```python
# Streamlit 기반 RAG 챗봇의 메인 진입점 및 전체 오케스트레이션을 담당하는 파일
"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

# [Lazy Import용] 런타임에 필요한 모듈들
...
import nest_asyncio
import streamlit as st
...
# 비동기 패치 적용 (Streamlit의 asyncio 루프 충돌 방지)
nest_asyncio.apply()
```

- [ ] **Step 3: Commit**

### Task 2: Restore Windows Integrity Logic

**Files:**
- Modify: `src/main.py:_check_windows_integrity`

- [ ] **Step 1: Restore the original logic including platform check and `torch` load test.**

```python
def _check_windows_integrity():
    """
    [Background] Windows 환경의 라이브러리 충돌을 체크하고 주기적으로 세션을 정리합니다.
    """
    try:
        from core.session import SessionManager
        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    except Exception as e:
        logger.error(f"[SYSTEM] [CLEANUP] 세션 정리 중 오류: {e}")

    # [최적화] CI 환경에서는 무거운 라이브러리 체크 생략 (충돌 위험 방지)
    import platform
    if platform.system() != "Windows" or os.getenv("GITHUB_ACTIONS") == "true":
        return

    try:
        # 무거운 라이브러리 로드 테스트 (핵심 RAG용)
        import sys
        if "torch" not in sys.modules:
            import torch
            _ = torch.tensor([1.0])
        logger.info("[SYSTEM] [INTEGRITY] Windows 라이브러리 무결성 점검 완료 (OK)")
    except Exception as e:
        logger.warning(f"[SYSTEM] [INTEGRITY] 점검 중 예외 발생: {e}")
```

- [ ] **Step 2: Commit**

### Task 3: Restore Upload Safety Checks

**Files:**
- Modify: `src/main.py:on_file_upload`

- [ ] **Step 1: Restore `MAX_FILE_SIZE_MB` validation and improve error messages.**

```python
def on_file_upload() -> None:
    ...
    # [개선] 파일 타입 검사 (MIME 타입 확인)
    if uploaded_file.type != "application/pdf":
        st.error("❌ 올바른 PDF 파일이 아닙니다. PDF 형식의 파일을 업로드해주세요.")
        return

    # [개선] 파일 크기 검사
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요."
        )
        return
    ...
```

- [ ] **Step 2: Commit**

### Task 4: Restore Missing Documentation and Comments

**Files:**
- Modify: `src/main.py` (various locations)

- [ ] **Step 1: Traverse the file and restore docstrings for functions that lost them.**
- [ ] **Step 2: Re-add explanatory comments for RAG rebuild and QA chain update logic.**

- [ ] **Step 3: Commit**

### Task 5: Verification

- [ ] **Step 1: Run the app and verify no crash on startup (verifies `nest_asyncio`).**
- [ ] **Step 2: Attempt to upload a large file (if possible) or mock it to verify `MAX_FILE_SIZE_MB` check.**
- [ ] **Step 3: Verify sidebar automation still works.**
