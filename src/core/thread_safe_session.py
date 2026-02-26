"""
Thread-Safe Session Management (Reliable Sync)

UI 스레드와 비동기 스레드 간의 완벽한 데이터 공유를 위해
전역 폴백 저장소를 주 데이터원으로 사용합니다.
"""

import builtins
import contextlib
import logging
import threading
import time
from contextvars import ContextVar
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

MAX_MESSAGE_HISTORY = 100
_session_id_var: ContextVar[str] = ContextVar("session_id", default="default")


class ThreadSafeSessionManager:
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
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "status_logs": [],
        "current_page": 1,
    }

    _fallback_sessions: dict[str, dict[str, Any]] = {}
    _global_lock = threading.RLock()

    @classmethod
    def get_session_id(cls) -> str:
        sid = _session_id_var.get()
        if sid != "default":
            return sid
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            ctx = get_script_run_ctx()
            if ctx:
                return ctx.session_id
        except Exception:
            pass
        return sid

    @classmethod
    def set_session_id(cls, session_id: str):
        _session_id_var.set(session_id)

    @classmethod
    def _get_state(cls, session_id: str | None = None) -> dict[str, Any]:
        """항상 전역 폴백 저장소를 반환합니다."""
        sid = session_id or cls.get_session_id()
        with cls._global_lock:
            if sid not in cls._fallback_sessions:
                cls._fallback_sessions[sid] = cls.DEFAULT_SESSION_STATE.copy()
                cls._fallback_sessions[sid]["messages"] = []
                cls._fallback_sessions[sid]["status_logs"] = []
            return cls._fallback_sessions[sid]

    @classmethod
    def init_session(cls, session_id: str | None = None):
        """세션을 초기화하고 필요한 경우 st.session_state를 업데이트합니다."""
        if session_id:
            cls.set_session_id(session_id)
        state = cls._get_state()

        # UI 스레드 동기화 (최소한의 필요한 값만)
        try:
            for k in [
                "pdf_processed",
                "pdf_file_path",
                "is_generating_answer",
                "new_file_uploaded",
            ]:
                if k in state:
                    st.session_state[k] = state[k]
        except Exception:
            pass

    @classmethod
    def get(cls, key: str, default: Any = None, session_id: str | None = None) -> Any:
        # 1. 먼저 폴백 저장소 확인
        state = cls._get_state(session_id)
        if key in state:
            return state[key]

        # 2. UI 스레드라면 st.session_state 확인
        try:
            return st.session_state.get(key, default)
        except Exception:
            return default

    @classmethod
    def set(cls, key: str, value: Any, session_id: str | None = None):
        # 1. 폴백 저장소 업데이트
        state = cls._get_state(session_id)
        with cls._global_lock:
            state[key] = value

        # 2. UI 스레드라면 즉시 반영
        with contextlib.suppress(builtins.BaseException):
            st.session_state[key] = value

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
        state = cls._get_state(session_id)
        msg = {
            "role": role,
            "content": content,
            "msg_type": msg_type,
            "timestamp": time.time(),
            **kwargs,
        }
        with cls._global_lock:
            current = list(state.get("messages", []))
            current.append(msg)
            state["messages"] = current[-MAX_MESSAGE_HISTORY:]
        with contextlib.suppress(builtins.BaseException):
            st.session_state["messages"] = state["messages"]

    @classmethod
    def add_status_log(cls, msg: str, session_id: str | None = None):
        state = cls._get_state(session_id)
        with cls._global_lock:
            current = list(state.get("status_logs", []))
            if current and current[-1] == msg:
                return
            current.append(msg)
            state["status_logs"] = current[-30:]
        with contextlib.suppress(builtins.BaseException):
            st.session_state["status_logs"] = state["status_logs"]
        cls.add_message("system", msg, msg_type="log", session_id=session_id)

    @classmethod
    def replace_last_status_log(cls, msg: str, session_id: str | None = None):
        state = cls._get_state(session_id)
        with cls._global_lock:
            logs = list(state.get("status_logs", []))
            if logs:
                logs[-1] = msg
            state["status_logs"] = logs
            msgs = list(state.get("messages", []))
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("msg_type") == "log":
                    msgs[i]["content"] = msg
                    break
            state["messages"] = msgs
        with contextlib.suppress(Exception):
            st.session_state["status_logs"] = state["status_logs"]
            st.session_state["messages"] = state["messages"]

    @classmethod
    def reset_all_state(cls, session_id: str | None = None):
        sid = session_id or cls.get_session_id()
        with cls._global_lock:
            if sid in cls._fallback_sessions:
                del cls._fallback_sessions[sid]
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
    def delete_session(cls, session_id: str) -> bool:
        with cls._global_lock:
            if session_id in cls._fallback_sessions:
                del cls._fallback_sessions[session_id]
                return True
        return False

    @classmethod
    def cleanup_expired_sessions(cls, max_idle_seconds: int = 3600):
        pass

    @classmethod
    def perform_security_audit(cls):
        """[Placeholder] 세션 보안 감사 수행"""
        pass

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """[Placeholder] 시스템 통계 반환"""
        return {
            "active_sessions": len(cls._fallback_sessions),
            "total_messages": sum(
                len(s.get("messages", [])) for s in cls._fallback_sessions.values()
            ),
        }
