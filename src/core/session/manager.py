# src/core/session/manager.py
"""
Session Management (Reliable Sync)

UI 스레드와 비동기 스레드 간의 완벽한 데이터 공유를 위해
전역 폴백 저장소를 주 데이터원으로 사용합니다.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
import uuid
from collections.abc import Set
from contextvars import ContextVar
from typing import Any

import streamlit as st

from common.constants import MAX_MESSAGE_HISTORY

logger = logging.getLogger(__name__)
_session_id_var: ContextVar[str] = ContextVar("session_id", default="default")
# 활성 세션 상한: 무분별한 세션 생성(예: API 헤더 남용)으로 인한 무한 성장 방지
MAX_ACTIVE_SESSIONS = 64


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
        "rebuild_done": False,
        "rebuild_error": None,
        "rebuild_status": None,
        "rebuild_progress": 0,
        "is_building_rag": False,
        "rebuild_cancelled": False,
        "rag_build_complete_flag": False,
        "status_logs": [],
        "current_page": 1,
    }

    _fallback_sessions: dict[str, dict[str, Any]] = {}
    _session_locks: dict[str, threading.Lock] = {}
    _map_lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        with cls._map_lock:
            cls._fallback_sessions.clear()
            cls._session_locks.clear()
        _session_id_var.set("default")

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
        with cls._map_lock:
            if sid not in cls._session_locks:
                cls._session_locks[sid] = threading.Lock()
            return cls._session_locks[sid]

    @classmethod
    def get_session_id(cls) -> str:
        """현재 컨텍스트의 세션 ID를 결정합니다.

        우선순위: 명시적으로 설정된 contextvar → Streamlit 세션 → "default".
        스레드 이름 기반의 임의 세션 생성은 제거했습니다 (교차 사용자 상태 누출 방지).
        세션이 필요한 코드는 반드시 session_id를 명시적으로 전달해야 합니다.
        """
        sid = _session_id_var.get()
        if sid and sid != "default":
            return sid

        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                ctx = get_script_run_ctx()
                if ctx and ctx.session_id:
                    _session_id_var.set(ctx.session_id)
                    return ctx.session_id
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.debug(
                    "[SESSION] Streamlit 컨텍스트 조회 실패, 폴백 사용: %s", exc
                )

        return "default"

    @classmethod
    def set_session_id(cls, session_id: str):
        if not session_id:
            return
        _session_id_var.set(session_id)

    @classmethod
    def _get_state(cls, session_id: str | None = None) -> dict[str, Any]:
        """항상 전역 폴백 저장소를 반환하며 접근 시간을 갱신합니다."""
        sid = session_id or cls.get_session_id()

        with cls._map_lock:
            if sid not in cls._fallback_sessions:
                if len(cls._fallback_sessions) >= MAX_ACTIVE_SESSIONS:
                    cls._evict_oldest_session_locked()
                new_state = cls.DEFAULT_SESSION_STATE.copy()
                new_state["messages"] = []
                new_state["status_logs"] = []
                new_state["_dirty_keys"] = set()
                new_state["last_accessed"] = time.time()
                cls._fallback_sessions[sid] = new_state
                logger.debug(f"[SESSION] 신규 세션 저장소 생성: {sid}")

            state = cls._fallback_sessions[sid]
            state["last_accessed"] = time.time()
            return state

    @classmethod
    def _evict_oldest_session_locked(cls) -> None:
        """가장 오래 사용되지 않은 세션을 퇴출합니다 (_map_lock 보유 중 호출).

        - 답변 생성/스트리밍 중인 세션(is_generating_answer)은 퇴출 대상에서
          제외합니다. 순수 LRU로 퇴출하면 스트리밍 세션이 도중에 사라집니다.
        - 잡혀 있는 세션 락을 pop하면 상호배제가 깨지므로, 해제된 락만
          정리합니다.
        """
        oldest_sid: str | None = None
        oldest_ts = float("inf")
        for candidate, state in cls._fallback_sessions.items():
            if candidate == "default":
                continue
            if state.get("is_generating_answer"):
                continue
            ts = state.get("last_accessed", 0)
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_sid = candidate
        if oldest_sid is not None:
            del cls._fallback_sessions[oldest_sid]
            lock = cls._session_locks.get(oldest_sid)
            if lock is None or not lock.locked():
                cls._session_locks.pop(oldest_sid, None)
            logger.warning(
                f"[SESSION] 활성 세션 상한({MAX_ACTIVE_SESSIONS}) 도달, "
                f"가장 오래된 세션 퇴출: {oldest_sid}"
            )

    @classmethod
    def init_session(cls, session_id: str | None = None):
        """세션을 초기화합니다. Streamlit 동기화는 sync_to_streamlit에서 수행합니다."""
        if session_id:
            cls.set_session_id(session_id)

        sid = session_id or cls.get_session_id()
        cls._get_state(sid)
        logger.debug(f"[SESSION] 세션 초기화 완료: {sid}")

    @classmethod
    def sync_to_streamlit(cls, session_id: str | None = None, key: str | None = None):
        """Streamlit UI 컨텍스트에서 세션 상태를 동기화합니다.
        선택적으로 특정 키만 동기화할 수 있습니다.
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

        # 상태를 _map_lock 안에서 취득 (퇴출/삭제 경합 제거)
        with cls._map_lock:
            state = cls._fallback_sessions.get(sid)
        if state is None:
            # 아직 없는 세션이면 생성 (기존 동작 유지)
            state = cls._get_state(sid)

        # 더티 키와 값을 세션 락 안에서 함께 스냅샷하여 TOCTOU, 부분 읽기,
        # 별칭(UI가 내부 리스트 공유) 문제를 제거합니다.
        values: dict[str, Any] = {}
        with cls._acquire_lock(sid):
            if key:
                if key in state:
                    values[key] = state[key]
            else:
                dirty = state["_dirty_keys"].copy()
                state["_dirty_keys"].clear()
                for k in dirty:
                    if k in state:
                        values[k] = state[k]

        if not values:
            return

        try:
            for k, val in values.items():
                if isinstance(val, list):
                    st.session_state[k] = val[:]
                elif isinstance(val, dict):
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
        """세션 상태에서 값을 가져옵니다.

        스레드 안전성: 세션별 락(_acquire_lock)을 전체 읽기 작업 동안
        유지하여 concurrent set()/add_message() 등으로부터 원자적 읽기를 보장합니다.
        잠금 순서: _acquire_lock(sid) → _global_lock (add_message/add_status_log과 일관).
        """
        sid = session_id or cls.get_session_id()

        if create:
            with cls._acquire_lock(sid):
                state = cls._get_state(sid)
                if key in state:
                    return state[key]
        else:
            with cls._acquire_lock(sid):
                fallback = cls._fallback_sessions.get(sid)
                if fallback is None:
                    return default
                if key in fallback:
                    return fallback[key]

        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                if get_script_run_ctx():
                    return st.session_state.get(key, default)
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.debug(
                    "[SESSION] Streamlit 세션 상태 조회 실패, 폴백 사용: %s", exc
                )
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
                logger.debug(f"[DEBUG] SessionManager.set: {sid} {k}={v}")
            state["_dirty_keys"].update(updates.keys())

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
        sid = session_id or cls.get_session_id()
        # Hold the per-session lock across the entire read-modify-write cycle
        # to prevent TOCTOU races with concurrent state["messages"] mutations.
        with cls._acquire_lock(sid):
            state = cls._get_state(sid)
            msg = {
                "msg_id": kwargs.pop("msg_id", str(uuid.uuid4())),
                "role": role,
                "content": content,
                "msg_type": msg_type,
                "timestamp": time.time(),
                **kwargs,
            }
            # Current list is mutated in-place (already referenced by state)
            current = state.get("messages", [])

            # Streaming message handling: if msg_id already exists, update it
            updated = False
            for i, existing_msg in enumerate(current):
                if existing_msg.get("msg_id") == msg["msg_id"]:
                    current[i].update(msg)
                    updated = True
                    break

            if not updated:
                current.append(msg)

            if len(current) > MAX_MESSAGE_HISTORY:
                # Trim creates a new list; assign it back to state
                state["messages"] = current[-MAX_MESSAGE_HISTORY:]
            elif not updated:
                state["messages"] = current

            state["_dirty_keys"].add("messages")

    @classmethod
    def add_status_log(
        cls, msg: str, session_id: str | None = None, add_to_chat: bool = False
    ):
        """시스템 상태 로그를 기록합니다. (UX 개편: 영구 대화 목록 오염 방지를 위해 기본값 False 변경)"""
        sid = session_id or cls.get_session_id()
        with cls._acquire_lock(sid):
            state = cls._get_state(sid)
            logs = state["status_logs"]
            if logs and logs[-1] == msg:
                return
            logs.append(msg)
            if len(logs) > 30:
                del logs[:-30]

            state["_dirty_keys"].add("status_logs")

        if add_to_chat:
            cls.add_message("system", msg, msg_type="log", session_id=session_id)

    @classmethod
    def replace_last_status_log(cls, msg: str, session_id: str | None = None):
        sid = session_id or cls.get_session_id()
        with cls._acquire_lock(sid):
            state = cls._get_state(sid)
            logs = state["status_logs"]
            if logs:
                logs[-1] = msg

            state["_dirty_keys"].add("status_logs")

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
            and not cls.get("is_building_rag", False, session_id=session_id)
            and not cls.get("needs_rag_rebuild", False, session_id=session_id)
            and not cls.get("needs_qa_chain_update", False, session_id=session_id)
        )

    @classmethod
    def reset_conversation(cls, session_id: str | None = None):
        """대화 상태(메시지 + chat_history 키)만 초기화합니다.

        문서/파이프라인/캐시 상태는 유지하며 채팅 타임라인만 비웁니다.
        RAG 그래프의 chat_history는 SessionManager의 messages에서 파생되므로
        (rag_core._get_recent_history) messages를 비우는 것으로 충분하며,
        만약 별도 chat_history 키가 남아 있으면 함께 제거합니다.
        """
        sid = session_id or cls.get_session_id()
        with cls._acquire_lock(sid):
            state = cls._get_state(sid)
            state["messages"] = []
            if "chat_history" in state:
                del state["chat_history"]
            state["_dirty_keys"].add("messages")

    @classmethod
    def reset_for_new_file(cls, session_id: str | None = None):
        # 새 문서 업로드 시 이전 문서에 대한 대화 내용도 함께 초기화합니다.
        cls.reset_conversation(session_id)
        cls.set("pdf_processed", False, session_id)
        cls.set("rag_engine", None, session_id)
        cls.set("file_hash", None, session_id)
        cls.set("pdf_processing_error", None, session_id)
        cls.set("rebuild_error", None, session_id)
        cls.set("rebuild_status", None, session_id)
        cls.set("rebuild_progress", 0, session_id)
        cls.set("rebuild_done", False, session_id)
        cls.set("rebuild_cancelled", False, session_id)
        cls.set("needs_rag_rebuild", False, session_id)
        cls.set("needs_qa_chain_update", False, session_id)
        cls.set("rag_build_complete_flag", False, session_id)
        cls.set("current_page", 1, session_id)
        cls.add_status_log("새 문서 분석 시작", session_id)

    @classmethod
    def safe_remove_file(cls, path: str, max_retries: int = 3):
        """[Windows 대응] 지수 백오프를 사용한 안전한 파일 삭제"""
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
        """세션을 삭제하고 무거운 객체의 참조를 명시적으로 해제합니다.

        물리적 파일 삭제는 _map_lock 밖에서 수행하여, Windows 파일 락
        백오프(sleep) 동안 전역 세션 접근이 블로킹되지 않도록 합니다.
        """
        pdf_path = None
        with cls._map_lock:
            if session_id not in cls._fallback_sessions:
                return False
            state = cls._fallback_sessions[session_id]

            # 물리적 파일 경로만 수집 (실제 삭제는 락 해제 후)
            pdf_path = state.get("pdf_file_path")

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
                # 잡히지 않은 락만 제거. 스트리밍 스레드가 잡고 있는 락을
                # pop하면 이후 동일 sid에 새 락이 생성되어 상호배제가 깨집니다.
                lock = cls._session_locks[session_id]
                if not lock.locked():
                    del cls._session_locks[session_id]

            logger.info(f"[SESSION] 세션 삭제 완료: {session_id}")

        if pdf_path:
            cls.safe_remove_file(pdf_path)
        return True

    @classmethod
    def cleanup_expired_sessions(cls, max_idle_seconds: int = 3600):
        """만료된 세션을 찾아 제거합니다. (딕셔너리 순회 중 삭제 방지)"""
        now = time.time()

        with cls._map_lock:
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
        with cls._map_lock:
            return {
                "active_sessions": len(cls._fallback_sessions),
                "total_messages": sum(
                    len(s.get("messages", [])) for s in cls._fallback_sessions.values()
                ),
            }

    @classmethod
    def get_active_file_hashes(cls) -> Set[str]:
        """현재 활성 세션들이 참조 중인 문서 해시 집합을 반환합니다."""
        with cls._map_lock:
            active: set[str] = set()
            for state in cls._fallback_sessions.values():
                file_hash = state.get("file_hash")
                if isinstance(file_hash, str) and file_hash:
                    active.add(file_hash)
            return active

    @classmethod
    def get_all_pdf_paths(cls) -> list[str]:
        """모든 활성 세션의 임시 PDF 경로를 반환합니다. (종료 핸들러용)"""
        with cls._map_lock:
            return [
                path
                for state in cls._fallback_sessions.values()
                if (path := state.get("pdf_file_path"))
            ]
