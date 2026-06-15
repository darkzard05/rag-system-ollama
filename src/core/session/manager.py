# src/core/session/manager.py
"""
Session Management (Reliable Sync)

UI 스레드와 비동기 스레드 간의 완벽한 데이터 공유를 위해
전역 폴백 저장소를 주 데이터원으로 사용합니다.
"""

import logging
import os
import threading
import time
from contextvars import ContextVar
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

MAX_MESSAGE_HISTORY = 100
_session_id_var: ContextVar[str] = ContextVar("session_id", default="default")


class SessionManager:
    """
    모든 데이터를 전역 딕셔너리에 저장하여 스레드 간 격리를 해제하고 동기화를 보장합니다.
    """

    DEFAULT_SESSION_STATE: dict[str, Any] = {
        "messages": [],
        "doc_pool": {},
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_selected_embedding_model": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_path": None,
        "file_hash": None,
        "rag_engine": None,
        "llm": None,
        "embedder": None,
        "is_generating_answer": False,
        "streaming_buffer": "",
        "streaming_thought": "",
        "global_status": "✅ 시스템 준비 완료",
        "status_level": "success",
        "global_progress": 0,
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "status_logs": [],
        "current_page": 1,
    }

    _fallback_sessions: dict[str, dict[str, Any]] = {}
    _session_locks: dict[str, threading.Lock] = {}
    _global_lock = threading.RLock()

    @classmethod
    def _is_streamlit_running(cls) -> bool:
        """Streamlit 런타임이 현재 스레드/프로세스에서 실행 중인지 확인합니다."""
        try:
            from streamlit.runtime import exists

            return exists()
        except ImportError:
            return False

    @classmethod
    def _acquire_lock(cls, session_id: str | None = None) -> threading.Lock:
        """세션별 전용 락을 반환합니다."""
        sid = session_id or cls.get_session_id()
        with cls._global_lock:
            if sid not in cls._session_locks:
                cls._session_locks[sid] = threading.Lock()
            return cls._session_locks[sid]

    @classmethod
    def get_session_id(cls) -> str:
        sid = _session_id_var.get()
        if sid and sid != "default":
            return sid

        # Streamlit 컨텍스트 확인 (최소화하여 오버헤드 감소)
        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                ctx = get_script_run_ctx()
                if ctx and ctx.session_id:
                    _session_id_var.set(ctx.session_id)
                    return ctx.session_id
            except Exception:
                pass

        return sid or "default"

    @classmethod
    def set_session_id(cls, session_id: str):
        if not session_id:
            return
        _session_id_var.set(session_id)

    @classmethod
    def _get_state(cls, session_id: str | None = None) -> dict[str, Any]:
        """항상 전역 폴백 저장소를 반환하며 접근 시간을 갱신합니다."""
        sid = session_id or cls.get_session_id()

        with cls._global_lock:
            if sid not in cls._fallback_sessions:
                new_state = cls.DEFAULT_SESSION_STATE.copy()
                new_state["messages"] = []
                new_state["status_logs"] = []
                new_state["last_accessed"] = time.time()
                cls._fallback_sessions[sid] = new_state
                logger.debug(f"[SESSION] 신규 세션 저장소 생성: {sid}")

            state = cls._fallback_sessions[sid]
            state["last_accessed"] = time.time()
            return state

    @classmethod
    def init_session(cls, session_id: str | None = None):
        """세션을 초기화합니다. Streamlit 동기화는 sync_to_streamlit에서 수행합니다."""
        if session_id:
            cls.set_session_id(session_id)

        sid = session_id or cls.get_session_id()
        cls._get_state(sid)
        logger.debug(f"[SESSION] 세션 초기화 완료: {sid}")

    @classmethod
    def sync_to_streamlit(cls, session_id: str | None = None):
        """Streamlit UI 컨텍스트에서 세션 상태를 동기화합니다.

        이 메서드는 UI 렌더링 단계에서만 호출되어야 하며,
        백그라운드 스레드에서 호출되면 무시됩니다.
        """
        if not cls._is_streamlit_running():
            return

        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            if not get_script_run_ctx():
                return
        except Exception:
            return

        sid = session_id or cls.get_session_id()
        state = cls._get_state(sid)

        # UI 스레드에서만 st.session_state 업데이트
        sync_keys = [
            "pdf_processed",
            "pdf_file_path",
            "is_generating_answer",
            "new_file_uploaded",
            "last_selected_model",
            "last_selected_embedding_model",
            "status_logs",
            "messages",
            "current_page",
        ]

        try:
            for k in sync_keys:
                if k in state:
                    # [수정] 참조 문제 방지를 위해 명시적으로 덮어쓰기 수행
                    val = state[k]
                    if isinstance(val, (list, dict)):
                        import copy

                        st.session_state[k] = copy.copy(val)
                    else:
                        st.session_state[k] = val
        except Exception as e:
            logger.warning(f"[SESSION] Streamlit 동기화 중 오류: {e}")

    @classmethod
    def get(
        cls,
        key: str,
        default: Any = None,
        session_id: str | None = None,
        create: bool = True,
    ) -> Any:
        """세션 상태에서 값을 가져옵니다."""
        if not create:
            sid = session_id or cls.get_session_id()
            with cls._global_lock:
                if sid not in cls._fallback_sessions:
                    return default
                state = cls._fallback_sessions[sid]
        else:
            state = cls._get_state(session_id)

        if key in state:
            return state[key]

        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                if get_script_run_ctx():
                    return st.session_state.get(key, default)
            except Exception:
                pass
        return default

    @classmethod
    def set(
        cls,
        key: str | None = None,
        value: Any = None,
        session_id: str | None = None,
        **kwargs,
    ):
        """단일 또는 다중 세션 데이터를 업데이트합니다."""
        sid = session_id or cls.get_session_id()
        state = cls._get_state(sid)

        updates = kwargs.copy()
        if key is not None:
            updates[key] = value

        with cls._acquire_lock(sid):
            for k, v in updates.items():
                state[k] = v

        # [수정] Streamlit 환경일 경우 st.session_state에도 즉시 반영하여 rerun 시 유실 방지
        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                if get_script_run_ctx():
                    for k, v in updates.items():
                        st.session_state[k] = v
            except Exception:
                pass

    @classmethod
    def delete(cls, key: str, session_id: str | None = None):
        """세션 상태에서 특정 키를 삭제합니다."""
        sid = session_id or cls.get_session_id()

        with cls._acquire_lock(sid):
            if sid in cls._fallback_sessions:
                state = cls._fallback_sessions[sid]
                if key in state:
                    del state[key]

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return cls.get("messages", [], session_id=session_id)

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        msg_type: str = "general",
        session_id: str | None = None,
        **kwargs,
    ):
        import uuid

        state = cls._get_state(session_id)
        msg = {
            "msg_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "msg_type": msg_type,
            "timestamp": time.time(),
            **kwargs,
        }

        with cls._acquire_lock(session_id or cls.get_session_id()):
            # [수정] 새로운 리스트를 생성하지 않고 기존 리스트를 가져와 업데이트
            current = state.get("messages", [])
            current.append(msg)
            if len(current) > MAX_MESSAGE_HISTORY:
                current = current[-MAX_MESSAGE_HISTORY:]

        # [수정] set 메서드를 호출하여 백그라운드와 st.session_state 양쪽 모두 동기화
        cls.set("messages", current, session_id=session_id)

    @classmethod
    def add_status_log(
        cls, msg: str, session_id: str | None = None, add_to_chat: bool = False
    ):
        """시스템 상태 로그를 기록합니다. (UX 개편: 영구 대화 목록 오염 방지를 위해 기본값 False 변경)"""
        state = cls._get_state(session_id)
        with cls._acquire_lock(session_id or cls.get_session_id()):
            current = list(state.get("status_logs", []))
            if current and current[-1] == msg:
                return
            current.append(msg)
            logs = current[-30:]

        cls.set("status_logs", logs, session_id=session_id)

        if add_to_chat:
            cls.add_message("system", msg, msg_type="log", session_id=session_id)

    @classmethod
    def replace_last_status_log(cls, msg: str, session_id: str | None = None):
        state = cls._get_state(session_id)
        sid = session_id or cls.get_session_id()
        with cls._acquire_lock(sid):
            logs = list(state.get("status_logs", []))
            if logs:
                logs[-1] = msg

        cls.set("status_logs", logs, session_id=sid)

    @classmethod
    def reset_all_state(cls, session_id: str | None = None):
        sid = session_id or cls.get_session_id()
        cls.delete_session(sid)
        cls.init_session(sid)

    @classmethod
    def is_ready_for_chat(cls, session_id: str | None = None) -> bool:
        return bool(
            cls.get("pdf_processed", session_id=session_id)
            and cls.get("rag_engine", session_id=session_id)
        )

    @classmethod
    def reset_for_new_file(cls, session_id: str | None = None):
        cls.set("pdf_processed", False, session_id)
        cls.set("rag_engine", None, session_id)
        cls.set("current_page", 1, session_id)
        cls.add_status_log("새 문서 분석 시작", session_id)

    @classmethod
    def safe_remove_file(cls, path: str, max_retries: int = 3):
        """[Windows 대응] 지수 백오프를 사용한 안전한 파일 삭제"""
        import time

        for attempt in range(max_retries):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"[SESSION] [CLEANUP] 파일 삭제 완료: {path}")
                return True
            except PermissionError:
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2**attempt)
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"[SESSION] [CLEANUP] 파일 삭제 최종 실패 (잠금): {path}"
                    )
            except Exception as e:
                logger.warning(f"[SESSION] [CLEANUP] 파일 삭제 중 예외 ({path}): {e}")
                break
        return False

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """세션을 삭제하고 무거운 객체의 참조를 명시적으로 해제합니다."""
        with cls._global_lock:
            if session_id in cls._fallback_sessions:
                state = cls._fallback_sessions[session_id]

                # 물리적 파일 삭제
                pdf_path = state.get("pdf_file_path")
                if pdf_path:
                    cls.safe_remove_file(pdf_path)

                # 무거운 객체 명시적 참조 해제 (메모리 누수 방지)
                heavy_keys = [
                    "rag_engine",
                    "llm",
                    "embedder",
                    "active_faiss_retriever",
                    "active_bm25_retriever",
                ]
                for k in heavy_keys:
                    if k in state:
                        state[k] = None

                del cls._fallback_sessions[session_id]

                if session_id in cls._session_locks:
                    del cls._session_locks[session_id]

                logger.info(f"[SESSION] 세션 삭제 완료: {session_id}")
                return True
        return False

    @classmethod
    def cleanup_expired_sessions(cls, max_idle_seconds: int = 3600):
        """만료된 세션을 찾아 제거합니다. (딕셔너리 순회 중 삭제 방지)"""
        now = time.time()

        with cls._global_lock:
            # 순회 중 딕셔너리 크기 변경을 막기 위해 키를 리스트로 복사
            expired_ids = [
                sid
                for sid, state in cls._fallback_sessions.items()
                if now - state.get("last_accessed", now) > max_idle_seconds
            ]

        if expired_ids:
            logger.info(
                f"[SYSTEM] [CLEANUP] {len(expired_ids)}개의 만료된 세션 정리 시작"
            )
            for sid in expired_ids:
                cls.delete_session(sid)

    @classmethod
    def perform_security_audit(cls):
        pass

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        with cls._global_lock:
            return {
                "active_sessions": len(cls._fallback_sessions),
                "total_messages": sum(
                    len(s.get("messages", [])) for s in cls._fallback_sessions.values()
                ),
            }
