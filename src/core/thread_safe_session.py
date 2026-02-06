"""
Thread-Safe Session Management

세션 상태 관리에 대한 thread-safe 구현입니다.
여러 스레드/비동기 작업에서 안전하게 세션 상태를 접근할 수 있습니다.
인스턴스 및 클래스 메서드 호출을 모두 지원합니다.
"""

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

from common.typing_utils import SessionData, SessionKey, SessionValue

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ✅ 메모리 누수 방지: 최대 메시지 히스토리
MAX_MESSAGE_HISTORY = 100

# [추가] 비동기 컨텍스트 격리를 위한 세션 ID 변수
_session_id_var: ContextVar[str] = ContextVar("session_id", default="default")


class ThreadSafeSessionManager:
    """
    Streamlit 세션 상태를 thread-safe하게 관리하는 클래스.
    """

    # [추가] 백그라운드 스레드용 전역 상태 (Streamlit Context 독립적)
    _is_generating_globally = False

    DEFAULT_SESSION_STATE: SessionData = {
        "messages": [],
        "doc_pool": {},  # 🚀 문서 중앙 저장소
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_selected_embedding_model": None,
        "last_pdf_name": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_path": None,
        "rag_engine": None,
        "vector_store": None,
        "llm": None,
        "embedder": None,
        "is_generating_answer": False,
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "status_logs": ["시스템 대기 중"],
        "current_embedding_device": "UNKNOWN",
        "current_page": 1,
    }

    # 클래스 레벨 속성 (공유 Lock 및 통계)
    _global_management_lock = threading.RLock()
    _session_locks: dict[str, threading.RLock] = {}
    _token_to_session_id: dict[str, str] = {}  # 🚀 토큰 -> 세션ID 매핑 추가
    _default_lock_timeout = 5.0
    lock_count = 0
    failed_acquisitions = 0
    _fallback_sessions: dict[
        str, Any
    ] = {}  # [수정] 단일 state에서 다중 세션 저장소로 변경

    def __init__(self, lock_timeout: float = 5.0):
        """인스턴스 기반 사용을 위한 초기화"""
        self.lock = threading.RLock()
        self.lock_timeout = lock_timeout
        self.lock_count = 0
        self.failed_acquisitions = 0

    @classmethod
    def _get_session_lock(cls, session_id: str) -> threading.RLock:
        """세션별 전용 락을 가져오거나 생성합니다."""
        with cls._global_management_lock:
            if session_id not in cls._session_locks:
                cls._session_locks[session_id] = threading.RLock()
            return cls._session_locks[session_id]

    @classmethod
    def _acquire_lock(cls, instance=None, session_id: str | None = None):
        """
        Lock 획득 context manager.
        session_id가 주어지면 해당 세션의 락을, 아니면 현재 컨텍스트의 락을 사용합니다.
        """
        if instance and hasattr(instance, "lock"):
            lock = instance.lock
            timeout = getattr(instance, "lock_timeout", cls._default_lock_timeout)
            target = instance
        else:
            sid = session_id or cls.get_session_id()
            lock = cls._get_session_lock(sid)
            timeout = cls._default_lock_timeout
            target = cls

        return _LockContext(lock, timeout, target)

    @classmethod
    def set_session_id(cls, session_id: str):
        """[추가] 현재 컨텍스트(스레드/태스크)에서 사용할 세션 ID를 설정합니다."""
        _session_id_var.set(session_id)

    @classmethod
    def get_session_id(cls) -> str:
        """현재 컨텍스트의 세션 ID를 가져옵니다. Streamlit 컨텍스트를 우선 확인합니다."""
        # 1. 먼저 ContextVar 확인
        sid = _session_id_var.get()

        # 2. ContextVar가 default면 Streamlit 컨텍스트 확인
        if sid == "default":
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                ctx = get_script_run_ctx()
                if ctx:
                    return ctx.session_id
            except (ImportError, Exception):
                pass
        return sid

    @classmethod
    def _get_state(cls, session_id: str | None = None):
        """
        세션 상태 저장소를 반환합니다.
        UI와 백그라운드 스레드 간 데이터 공유를 위해 _fallback_sessions를 주 저장소로 사용합니다.
        """
        sid = session_id or cls.get_session_id()

        # 관리 락을 사용하여 세션 저장소 접근 보호 (매우 짧은 범위)
        with cls._global_management_lock:
            if sid not in cls._fallback_sessions:
                # 새로운 세션 상태 초기화
                new_state = cls.DEFAULT_SESSION_STATE.copy()
                # 가변 객체들은 새로 생성
                new_state["messages"] = []
                new_state["doc_pool"] = {}
                new_state["status_logs"] = list(
                    cls.DEFAULT_SESSION_STATE["status_logs"]
                )
                new_state["_last_activity"] = time.time()
                new_state["_initialized"] = True
                cls._fallback_sessions[sid] = new_state

            # 활동 시간 업데이트
            cls._fallback_sessions[sid]["_last_activity"] = time.time()
            return cls._fallback_sessions[sid]

    @classmethod
    def cleanup_expired_sessions(cls, max_idle_seconds: int = 3600):
        """[추가] 일정 시간 동안 활동이 없는 세션을 삭제하여 메모리를 확보합니다."""
        now = time.time()
        expired_ids = []

        with cls._global_management_lock:
            for sid, state in cls._fallback_sessions.items():
                if sid == "default":
                    continue
                last_activity = state.get("_last_activity", 0)
                if now - last_activity > max_idle_seconds:
                    expired_ids.append(sid)

        for sid in expired_ids:
            # 락 획득 순서: Session Lock -> Global Lock (delete_session 내부에서 지킴)
            cls.delete_session(sid)

        if expired_ids:
            logger.info(
                f"[SYSTEM] [SESSION] 만료된 세션 삭제 완료 | 개수: {len(expired_ids)}"
            )

    @classmethod
    def init_session(cls, session_id: str | None = None):
        if session_id:
            cls.set_session_id(session_id)

        # _get_state 내부에서 이미 초기화를 수행하므로 여기서는 락만 걸어 확인
        with cls._acquire_lock():
            state = cls._get_state()
            if not state.get("_initialized", False):
                state["_initialized"] = True

    @classmethod
    def get(
        cls,
        key: str,
        default: SessionValue | None = None,
        session_id: str | None = None,
    ) -> SessionValue | None:
        with cls._acquire_lock(session_id=session_id):
            return cls._get_state(session_id=session_id).get(key, default)

    @classmethod
    def set(
        cls, key: SessionKey, value: SessionValue, session_id: str | None = None
    ) -> None:
        with cls._acquire_lock(session_id=session_id):
            cls._get_state(session_id=session_id)[key] = value
            if key == "is_generating_answer":
                cls._is_generating_globally = bool(value)

    def set_inst(self, key: str, value: Any, session_id: str | None = None) -> None:
        with self._acquire_lock(self, session_id=session_id):
            self._get_state(session_id=session_id)[key] = value

    @classmethod
    def has_key(cls, key: str, session_id: str | None = None) -> bool:
        with cls._acquire_lock(session_id=session_id):
            return key in cls._get_state(session_id=session_id)

    def exists(self, key: str, session_id: str | None = None) -> bool:
        with self._acquire_lock(self, session_id=session_id):
            return key in self._get_state(session_id=session_id)

    @classmethod
    def delete_key(cls, key: str, session_id: str | None = None) -> bool:
        with cls._acquire_lock(session_id=session_id):
            state = cls._get_state(session_id=session_id)
            if key in state:
                del state[key]
                return True
            return False

    def delete(self, key: str, session_id: str | None = None) -> bool:
        return self.delete_key(key, session_id=session_id)

    @classmethod
    def clear_all(cls, session_id: str | None = None):
        with cls._acquire_lock(session_id=session_id):
            cls._get_state(session_id=session_id).clear()

    def clear(self, session_id: str | None = None) -> None:
        self.clear_all(session_id=session_id)

    @classmethod
    def atomic_read(
        cls, keys: list[str], session_id: str | None = None
    ) -> dict[str, Any]:
        with cls._acquire_lock(session_id=session_id):
            state = cls._get_state(session_id=session_id)
            return {key: state.get(key) for key in keys}

    @classmethod
    def atomic_update(
        cls,
        update_func: Callable[[dict[str, Any]], dict[str, Any]],
        session_id: str | None = None,
    ) -> bool:
        with cls._acquire_lock(session_id=session_id):
            try:
                state = cls._get_state(session_id=session_id)
                updates = update_func(dict(state))
                for key, value in updates.items():
                    state[key] = value
                return True
            except Exception as e:
                logger.error(f"Atomic update 실패: {e}")
                return False

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """[추가] 특정 세션을 메모리 저장소에서 완전히 삭제합니다."""
        # 명시적으로 해당 세션의 락을 획득
        with cls._acquire_lock(session_id=session_id), cls._global_management_lock:
            if session_id in cls._fallback_sessions:
                session_data = cls._fallback_sessions[session_id]

                # 리소스 명시적 해제
                vs = session_data.get("vector_store")
                if vs and hasattr(vs, "index") and hasattr(vs.index, "reset"):
                    with contextlib.suppress(Exception):
                        vs.index.reset()

                session_data.clear()
                del cls._fallback_sessions[session_id]

                # 락도 제거
                if session_id in cls._session_locks:
                    del cls._session_locks[session_id]

                logger.info(
                    f"[SYSTEM] [SESSION] 세션 데이터 삭제 완료 | ID: {session_id}"
                )
                return True

            return False

    @classmethod
    def get_all_state(cls, session_id: str | None = None) -> dict[str, Any]:
        """[최적화] 현재 세션의 모든 상태를 반환합니다 (참조 반환으로 오버헤드 최소화)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            # 외부 수정을 방지하기 위해 얕은 복사만 수행
            return cls._get_state(session_id=session_id).copy()

    @classmethod
    def get_stats(cls, session_id: str | None = None) -> dict[str, Any]:
        target = cls
        session_keys = 0
        with contextlib.suppress(Exception):
            session_keys = len(cls._get_state(session_id=session_id))

        return {
            "lock_acquisitions": target.lock_count,
            "failed_acquisitions": target.failed_acquisitions,
            "session_keys": session_keys,
        }

    def set_multiple(self, data: dict[str, Any], session_id: str | None = None) -> bool:
        with self._acquire_lock(self, session_id=session_id):
            state = self._get_state(session_id=session_id)
            for key, value in data.items():
                state[key] = value
            return True

    def get_multiple(
        self, keys: list[str], session_id: str | None = None
    ) -> dict[str, Any]:
        return self.atomic_read(keys, session_id=session_id)

    def reset_stats(self) -> None:
        self.lock_count = 0
        self.failed_acquisitions = 0

    def is_healthy(self) -> bool:
        return self.failed_acquisitions == 0

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, str]]:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            return cls._get_state(session_id=session_id).get("messages", []).copy()

    @classmethod
    def reset_all_state(cls, session_id: str | None = None):
        """[최적화] 모든 세션 상태를 기본값으로 효율적으로 리셋합니다."""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            logger.debug(
                f"[Session] [Reset] 세션 상태 리셋 (ID: {session_id or 'current'})"
            )
            state = cls._get_state(session_id=session_id)
            # 1. 기존 상태 초기화
            state.clear()
            # 2. 기본값 일괄 적용 (업데이트 방식으로 성능 향상)
            state.update(cls.DEFAULT_SESSION_STATE)
            # 3. 가변 객체 깊은 복사 (필요한 것만)
            state["messages"] = []
            state["doc_pool"] = {}
            state["status_logs"] = list(
                cls.DEFAULT_SESSION_STATE.get("status_logs", [])
            )
            state["_initialized"] = True
            state["_last_activity"] = time.time()

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        processed_content: str | None = None,
        msg_type: str = "general",
        session_id: str | None = None,
        status_logs: list[str] | None = None,
        **kwargs,
    ):
        """[최적화] 세션에 새로운 메시지를 추가합니다. (ChatMessage 스키마 적용)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            state = cls._get_state(session_id=session_id)
            if "messages" not in state:
                state["messages"] = []
            if "doc_pool" not in state:
                state["doc_pool"] = {}

            # [최적화] 문서 객체가 있으면 풀링 처리
            documents = kwargs.get("documents")
            if documents:
                from common.utils import fast_hash

                doc_ids = []
                for doc in documents:
                    doc_key = f"{doc.page_content}_{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
                    content_hash = fast_hash(doc_key)
                    if content_hash not in state["doc_pool"]:
                        state["doc_pool"][content_hash] = doc
                    doc_ids.append(content_hash)
                kwargs["doc_ids"] = doc_ids
                # documents 인자는 pydantic ChatMessage 스키마에 없으므로 제거
                if "documents" in kwargs:
                    del kwargs["documents"]

            # [핵심] ChatMessage 스키마 기반 데이터 생성
            from api.schemas import ChatMessage

            # status_logs가 없으면 현재 세션의 최신 로그를 사용 (선택 사항)
            final_logs = status_logs
            if not final_logs and role == "assistant":
                # 현재 세션의 로그 중 '시스템 대기 중' 이후의 것들만 캡처
                all_logs = state.get("status_logs", [])
                final_logs = all_logs.copy() if all_logs else None

            message_obj = ChatMessage(
                role=role,
                content=content,
                processed_content=processed_content,
                msg_type=msg_type,
                thought=kwargs.get("thought"),
                doc_ids=kwargs.get("doc_ids", []),
                metrics=kwargs.get("metrics"),
                timestamp=time.time(),
            )

            # ChatMessage 스키마에 status_logs가 없을 수 있으므로 dict 업데이트로 보완
            msg_dict = message_obj.model_dump()
            if final_logs:
                msg_dict["status_logs"] = final_logs

            state["messages"].append(msg_dict)
            state["_last_activity"] = time.time()

    @classmethod
    def is_ready_for_chat(cls, session_id: str | None = None) -> bool:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            state = cls._get_state(session_id=session_id)
            # 캐시 로직 재도입
            if not state.get("_chat_ready_needs_refresh", True):
                return state.get("_cached_chat_ready", False)

            result = (
                state.get("pdf_processed", False)
                and not state.get("pdf_processing_error")
                and state.get("rag_engine") is not None
            )

            state["_cached_chat_ready"] = result
            state["_chat_ready_needs_refresh"] = False
            return result

    @classmethod
    def reset_for_new_file(cls, session_id: str | None = None):
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            logger.debug(
                f"[SYSTEM] [EVENT] 새 파일 업로드 감지 (ID: {session_id or 'current'}) | RAG 상태 리셋"
            )
            state = cls._get_state(session_id=session_id)

            # [최적화] 이전 벡터 저장소 메모리 명시적 해제 (VRAM 누수 방지)
            old_vs = state.get("vector_store")
            if old_vs:
                try:
                    if hasattr(old_vs, "index"):
                        old_vs.index.reset()
                    del old_vs

                    import gc

                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(
                        "[Session] [Cleanup] FAISS 인덱스 및 VRAM 메모리 해제 완료"
                    )
                except Exception as e:
                    logger.warning(f"FAISS 메모리 해제 실패: {e}")

            keys_to_reset = [
                "pdf_processed",
                "pdf_processing_error",
                "rag_engine",
                "vector_store",
            ]
            for key in keys_to_reset:
                if key in state:
                    state[key] = None
            state["pdf_processed"] = False
            state["needs_rag_rebuild"] = True
            state["_chat_ready_needs_refresh"] = True
            state["current_page"] = 1  # 새 파일 업로드 시 1페이지로 리셋

            # [수정] 구분선 없이 새 작업 알림만 추가
            if "status_logs" not in state:
                state["status_logs"] = []

            start_msg = "새 문서 분석 시작"
            if not state["status_logs"] or state["status_logs"][-1] != start_msg:
                state["status_logs"].append(start_msg)

            # 최신 30개 유지 정책 적용
            if len(state["status_logs"]) > 30:
                state["status_logs"] = state["status_logs"][-30:]

    @classmethod
    def add_status_log(cls, msg: str, session_id: str | None = None):
        """작업 로그를 추가합니다. (최신 30개 보관)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            state = cls._get_state(session_id=session_id)
            if "status_logs" not in state:
                state["status_logs"] = []

            if state["status_logs"] and state["status_logs"][-1] == msg:
                return

            state["status_logs"].append(msg)
            # [수정] 히스토리 유지 개수 상향 (10 -> 30)
            if len(state["status_logs"]) > 30:
                state["status_logs"] = state["status_logs"][-30:]

    @classmethod
    def replace_last_status_log(cls, msg: str, session_id: str | None = None):
        """가장 최근 로그를 새로운 메시지로 교체합니다. (진행 상태 업데이트용)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(
            instance=target, session_id=session_id
        ):
            state = cls._get_state(session_id=session_id)
            if "status_logs" not in state or not state["status_logs"]:
                state["status_logs"] = [msg]
            else:
                state["status_logs"][-1] = msg


class _LockContext:
    def __init__(self, lock, timeout, target):
        self.lock = lock
        self.timeout = timeout
        self.target = target
        self.acquired = False

    def __enter__(self):
        # [개선] 현재 실행 중인 스레드가 이벤트 루프 스레드인지 확인 (FastAPI 대응)
        try:
            import asyncio

            is_in_loop = False
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop()
                is_in_loop = True
        except ImportError:
            is_in_loop = False

        # 이벤트 루프 스레드라면 아주 짧은 타임아웃으로 시도하고, 실패 시 즉시 양보하도록 설계
        # (실제 완벽한 비동기 락은 아니지만 루프 프리징을 최소화함)
        actual_timeout = 0.1 if is_in_loop else self.timeout

        self.acquired = self.lock.acquire(timeout=actual_timeout)

        if not self.acquired:
            # 이벤트 루프에서 0.1초 내에 획득 실패 시,
            # 일반적인 동기 스레드와 달리 루프 보호를 위해 즉시 에러 발생 또는 재시도 로직 유도
            from common.exceptions import SessionLockTimeoutError

            self.target.failed_acquisitions += 1

            error_msg = (
                "이벤트 루프 보호를 위해 세션 락 획득이 거부되었습니다."
                if is_in_loop
                else f"{self.timeout}초 내에 세션 락을 획득하지 못했습니다."
            )

            raise SessionLockTimeoutError(
                error_msg,
                details={
                    "timeout": actual_timeout,
                    "is_event_loop": is_in_loop,
                    "target_type": type(self.target).__name__,
                    "active_threads": threading.active_count(),
                },
            )

        self.target.lock_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.lock.release()
