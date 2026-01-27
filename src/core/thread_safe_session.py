"""
Thread-Safe Session Management

세션 상태 관리에 대한 thread-safe 구현입니다.
여러 스레드/비동기 작업에서 안전하게 세션 상태를 접근할 수 있습니다.
인스턴스 및 클래스 메서드 호출을 모두 지원합니다.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, TypeVar, Any, Callable

import streamlit as st
from common.typing_utils import SessionData, SessionKey, SessionValue

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ✅ 메모리 누수 방지: 최대 메시지 히스토리
MAX_MESSAGE_HISTORY = 1000


class ThreadSafeSessionManager:
    """
    Streamlit 세션 상태를 thread-safe하게 관리하는 클래스.
    """
    
    # [추가] 백그라운드 스레드용 전역 상태 (Streamlit Context 독립적)
    _is_generating_globally = False

    DEFAULT_SESSION_STATE: SessionData = {
        "messages": [],
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
        "pdf_interaction_blocked": False,
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "show_graph": False,
        "status_logs": ["시스템 대기 중", "PDF 업로드 필요"], 
    }

    # 클래스 레벨 속성 (공유 Lock 및 통계)
    _global_lock = threading.RLock()
    _default_lock_timeout = 5.0
    lock_count = 0
    failed_acquisitions = 0
    _fallback_state = {} # 폴백 저장소 초기화

    def __init__(self, lock_timeout: float = 5.0):
        """인스턴스 기반 사용을 위한 초기화"""
        self.lock = threading.RLock()
        self.lock_timeout = lock_timeout
        self.lock_count = 0
        self.failed_acquisitions = 0

    @classmethod
    def _acquire_lock(cls, instance=None):
        """Lock 획득 context manager"""
        lock = instance.lock if instance else cls._global_lock
        timeout = instance.lock_timeout if instance else cls._default_lock_timeout
        target = instance if instance else cls
        return _LockContext(lock, timeout, target)

    @classmethod
    def _get_state(cls):
        """Streamlit session_state 또는 폴백 딕셔너리를 반환합니다."""
        try:
            # Streamlit 컨텍스트 확인 (접근 시도)
            _ = st.session_state
            return st.session_state
        except Exception:
            # 컨텍스트가 없는 경우(API 서버 등)를 위한 전역 저장소 사용
            if not hasattr(cls, "_fallback_state") or not cls._fallback_state:
                cls._fallback_state = cls.DEFAULT_SESSION_STATE.copy()
            return cls._fallback_state

    @classmethod
    def _get_state(cls):
        """Streamlit session_state 또는 폴백 딕셔너리를 반환합니다."""
        try:
            # Streamlit 컨텍스트 확인
            return st.session_state
        except Exception:
            # 컨텍스트가 없는 경우(API 서버 등)를 위한 전역 저장소 사용
            if not hasattr(cls, "_fallback_state"):
                cls._fallback_state = cls.DEFAULT_SESSION_STATE.copy()
            return cls._fallback_state

    @classmethod
    def init_session(cls):
        with cls._acquire_lock():
            state = cls._get_state()
            if not state.get("_initialized", False):
                logger.info("[Session] [Init] 세션 상태 초기화 중...")
                for key, value in cls.DEFAULT_SESSION_STATE.items():
                    if key not in state:
                        state[key] = value
                if hasattr(state, "_initialized"):
                    state._initialized = True
                else:
                    state["_initialized"] = True

    @classmethod
    def get(cls, key: str, default: Optional[SessionValue] = None) -> Optional[SessionValue]:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            return cls._get_state().get(key, default)

    @classmethod
    def set(cls, key: SessionKey, value: SessionValue) -> None:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            cls._get_state()[key] = value
            # 전역 플래그 동기화
            if key == "is_generating_answer":
                ThreadSafeSessionManager._is_generating_globally = bool(value)

    def set_inst(self, key: str, value: Any) -> None:
        """인스턴스 메서드용 set (테스트 호환성)"""
        with self._acquire_lock(self):
            self._get_state()[key] = value

    @classmethod
    def has_key(cls, key: str) -> bool:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            return key in cls._get_state()

    def exists(self, key: str) -> bool:
        """인스턴스 메서드용 exists (테스트 호환성)"""
        with self._acquire_lock(self):
            return key in self._get_state()

    @classmethod
    def delete_key(cls, key: str) -> bool:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
            if key in state:
                del state[key]
                return True
            return False

    def delete(self, key: str) -> bool:
        """인스턴스 메서드용 delete (테스트 호환성)"""
        return self.delete_key(key)

    @classmethod
    def clear_all(cls):
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            cls._get_state().clear()
            logger.info("[Session] [Cleanup] 모든 세션 상태 삭제")

    def clear(self) -> None:
        """인스턴스 메서드용 clear (테스트 호환성)"""
        self.clear_all()

    @classmethod
    def atomic_read(cls, keys: List[str]) -> Dict[str, Any]:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
            return {key: state.get(key) for key in keys}

    @classmethod
    def atomic_update(cls, update_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> bool:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            try:
                state = cls._get_state()
                current_state = dict(state)
                updates = update_func(current_state)
                for key, value in updates.items():
                    state[key] = value
                return True
            except Exception as e:
                logger.error(f"Atomic update 실패: {e}")
                return False

    @classmethod
    def get_all_state(cls) -> Dict[str, Any]:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            return dict(cls._get_state())

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        target = cls
        session_keys = 0
        try:
            session_keys = len(cls._get_state())
        except Exception:
            pass
            
        return {
            "lock_acquisitions": target.lock_count,
            "failed_acquisitions": target.failed_acquisitions,
            "session_keys": session_keys,
        }

    def set_multiple(self, data: Dict[str, Any]) -> bool:
        with self._acquire_lock(self):
            state = self._get_state()
            for key, value in data.items():
                state[key] = value
            return True

    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        return self.atomic_read(keys)

    def reset_stats(self) -> None:
        self.lock_count = 0
        self.failed_acquisitions = 0

    def is_healthy(self) -> bool:
        return self.failed_acquisitions == 0

    @classmethod
    def get_messages(cls) -> List[Dict[str, str]]:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            return cls._get_state().get("messages", []).copy()

    @classmethod
    def reset_all_state(cls):
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            logger.info("[Session] [Reset] 모든 세션 상태 리셋")
            state = cls._get_state()
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                state[key] = value
            if hasattr(state, "_initialized"):
                state._initialized = True
            else:
                state["_initialized"] = True

    @classmethod
    def add_message(cls, role: str, content: str, **kwargs):
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
            if "messages" not in state:
                state["messages"] = []
            
            msg = {"role": role, "content": content}
            msg.update(kwargs)
            state["messages"].append(msg)
            
            if len(state["messages"]) > MAX_MESSAGE_HISTORY:
                state["messages"] = state["messages"][-MAX_MESSAGE_HISTORY:]

    @classmethod
    def is_ready_for_chat(cls) -> bool:
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
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
    def reset_for_new_file(cls):
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            logger.info("[Session] [Event] 새 파일 업로드 감지 -> RAG 상태 리셋")
            state = cls._get_state()
            keys_to_reset = ["pdf_processed", "pdf_processing_error", "rag_engine", "vector_store"]
            for key in keys_to_reset:
                if key in state:
                    state[key] = None
            state["pdf_processed"] = False
            state["needs_rag_rebuild"] = True
            state["_chat_ready_needs_refresh"] = True
            # [수정] 이전 로그를 완전히 비우고 분석 시작 상태만 표시
            state["status_logs"] = ["문서 분석 중"]

    @classmethod
    def add_status_log(cls, msg: str):
        """작업 로그를 추가합니다. (내부 10개 보관, UI는 최신 4개 표시)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
            if "status_logs" not in state:
                state["status_logs"] = []
            
            if state["status_logs"] and state["status_logs"][-1] == msg:
                return
                
            state["status_logs"].append(msg)
            # 최신 10개 유지 (스크롤 효과를 위한 버퍼)
            if len(state["status_logs"]) > 10:
                state["status_logs"] = state["status_logs"][-10:]

    @classmethod
    def replace_last_status_log(cls, msg: str):
        """가장 최근 로그를 새로운 메시지로 교체합니다. (진행 상태 업데이트용)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
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
        self.acquired = self.lock.acquire(timeout=self.timeout)
        if not self.acquired:
            self.target.failed_acquisitions += 1
            self.lock.acquire()
            self.acquired = True
        
        self.target.lock_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.lock.release()