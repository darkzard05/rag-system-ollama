# Remove Monkey Patches and Implement Clean Worker Pattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `nest_asyncio` dependencies and implement a robust background worker utility for isolated heavy tasks.

**Architecture:** Moving from nested event loops to a clean "one thread per worker" model using `asyncio.run()` and Streamlit's `add_script_run_ctx` for context propagation.

**Tech Stack:** Python, Asyncio, Streamlit, Threading.

---

### Task 1: Implement `run_in_background_worker` in `src/common/utils.py`

**Files:**
- Modify: `src/common/utils.py`

- [ ] **Step 1: Add the utility function**

Add `run_in_background_worker` to the end of `src/common/utils.py`.

```python
def run_in_background_worker(coro, session_id: str):
    """
    백그라운드 스레드에서 비동기 작업을 실행하고 완료 시 Streamlit 앱을 리런합니다.
    nest_asyncio 없이도 안전하게 비동기 작업을 실행할 수 있도록 설계되었습니다.
    """
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    from streamlit.runtime import get_instance
    import threading
    import asyncio
    from core.session import SessionManager

    # 현재 메인 스레드의 Streamlit 컨텍스트 획득
    ctx = get_script_run_ctx()

    def _wrapper():
        # 1. 새 스레드에 Streamlit 컨텍스트 주입
        add_script_run_ctx(threading.current_thread(), ctx)
        
        # 2. 스레드 로컬 세션 ID 설정 (멀티유저 격리)
        SessionManager.set_session_id(session_id)
        
        # 3. 독립된 이벤트 루프에서 코루틴 실행
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.error(f"[WORKER] 백그라운드 작업 중 오류 발생: {e}", exc_info=True)
        finally:
            # 4. 작업 완료 후 해당 세션 리런 요청
            instance = get_instance()
            if instance:
                instance.request_rerun(session_id)
                logger.info(f"[WORKER] 세션 {session_id} 리런 요청 완료")

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    return thread
```

- [ ] **Step 2: Update `sync_run` to be more descriptive**

Modify the docstring of `sync_run` since `nest_asyncio` will be removed.

```python
def sync_run(coro):
    """
    Streamlit(동기 환경)에서 비동기 코루틴을 안전하게 실행하기 위한 헬퍼.
    주의: 이미 실행 중인 이벤트 루프 내부에서 호출하면 안 됩니다.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # nest_asyncio가 적용된 환경(기존)에서는 작동하지만, 
        # 제거된 환경에서는 RuntimeError를 발생시킬 수 있습니다.
        return loop.run_until_complete(coro)

    return asyncio.run(coro)
```

### Task 2: Remove Monkey Patches from `src/main.py`

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Remove `nest_asyncio` imports and initialization**

Remove:
```python
import nest_asyncio
...
nest_asyncio.apply()
```

- [ ] **Step 2: Remove `_patch_current_task_for_nest_asyncio` function and its call**

Remove lines 33-54.

### Task 3: Refactor `_bg_rebuild_thread` and usage in `src/main.py`

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Convert `_bg_rebuild_thread` to an async task**

Rename `_bg_rebuild_thread` to `_bg_rebuild_task` and change signature to `async def`. Remove internal `async def _run()` and `asyncio.run()`.

```python
async def _bg_rebuild_task(session_id: str, file_path: str, file_name: str, embedder_name: str):
    """
    백그라운드에서 실행될 RAG 빌드 비동기 태스크.
    run_in_background_worker에 의해 실행됩니다.
    """
    from core.model_loader import ModelManager
    from core.rag_core import RAGOrchestrator
    from core.session import SessionManager

    # run_in_background_worker가 이미 SessionManager.set_session_id(session_id)를 호출했지만,
    # 명시적 보장을 위해 한 번 더 수행할 수 있습니다.
    SessionManager.set_session_id(session_id)

    SessionManager.set("rebuild_done", False)
    SessionManager.set("rebuild_error", None)
    SessionManager.set("rebuild_status", f"'{file_name}' 분석 중...")

    try:
        embedder = await ModelManager.get_embedder(embedder_name)
        rag_sys = RAGOrchestrator(session_id=session_id)

        # 파이프라인 빌드
        success_message, cache_used = await rag_sys.build_pipeline(
            file_path=file_path, file_name=file_name, embedder=embedder
        )

        SessionManager.set("pdf_processed", True)
        SessionManager.add_status_log(f"✅ {success_message}")
        SessionManager.add_message("system", success_message)
        SessionManager.add_message("system", "READY_FOR_QUERY")
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("rebuild_error", error_msg)
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.set("pdf_processed", True)
        SessionManager.add_message("system", f"❌ {error_msg}")
    finally:
        SessionManager.set("rebuild_done", True)
        SessionManager.set("is_building_rag", False)
```

- [ ] **Step 2: Update `_handle_pending_tasks` to use the new worker**

Import `run_in_background_worker` from `common.utils`.
Replace manual thread creation with `run_in_background_worker`.

Locations:
1. Near line 334 (New file upload)
2. Near line 353 (RAG rebuild request)

```python
        from common.utils import run_in_background_worker
        run_in_background_worker(
            _bg_rebuild_task(current_sid, current_file_path, current_file_name, current_embedding_model),
            current_sid
        )
```

### Task 4: Verification

- [ ] **Step 1: Check for syntax errors**

Run `python -m py_compile src/main.py src/common/utils.py`

- [ ] **Step 2: Verify PDF upload flow**

Since I cannot run Streamlit visually, I will create a small script that simulates the background worker and verifies it triggers a "rerun" (mocked).

```python
# scripts/verification/test_background_worker.py
import asyncio
import threading
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

async def mock_coro():
    print("Coro running...")
    await asyncio.sleep(0.1)
    print("Coro done.")

def test_worker():
    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx, \
         patch("streamlit.runtime.get_instance") as mock_get_instance:
        
        mock_instance = MagicMock()
        mock_get_instance.return_value = mock_instance
        
        from common.utils import run_in_background_worker
        
        session_id = "test_session"
        thread = run_in_background_worker(mock_coro(), session_id)
        thread.join()
        
        mock_instance.request_rerun.assert_called_once_with(session_id)
        print("Verification SUCCESS: Worker executed and requested rerun.")

if __name__ == "__main__":
    test_worker()
```
