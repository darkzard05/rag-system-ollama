"""
Thread-Safe Session Management

세션 상태 관리에 대한 thread-safe 구현입니다.
여러 스레드/비동기 작업에서 안전하게 세션 상태를 접근할 수 있습니다.
인스턴스 및 클래스 메서드 호출을 모두 지원합니다.
"""

import logging
import threading
import time
import copy
import hashlib
from typing import Dict, List, Optional, TypeVar, Any, Callable
from contextvars import ContextVar

import streamlit as st
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
        "doc_pool": {},  # 🚀 문서 중앙 저장소 (메모리 절감용)
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
    _fallback_sessions = {} # [수정] 단일 state에서 다중 세션 저장소로 변경

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
    def set_session_id(cls, session_id: str):
        """[추가] 현재 컨텍스트(스레드/태스크)에서 사용할 세션 ID를 설정합니다."""
        _session_id_var.set(session_id)

    @classmethod
    def get_session_id(cls) -> str:
        """[추가] 현재 컨텍스트의 세션 ID를 가져옵니다."""
        return _session_id_var.get()

    @classmethod
    def _get_state(cls):
        """Streamlit session_state 또는 폴백 딕셔너리를 반환합니다."""
        try:
            # Streamlit 컨텍스트 확인 (runtime 체크)
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                return st.session_state
            
            # 컨텍스트가 없는 경우(API 서버 등)를 위한 세션별 저장소 사용
            sid = cls.get_session_id()
            if sid not in cls._fallback_sessions:
                cls._fallback_sessions[sid] = copy.deepcopy(cls.DEFAULT_SESSION_STATE)
                cls._fallback_sessions[sid]["_initialized"] = True
            return cls._fallback_sessions[sid]
        except (Exception, ImportError):
            # 라이브러리 부재 시에도 세션별 저장소 사용
            sid = cls.get_session_id()
            if sid not in cls._fallback_sessions:
                cls._fallback_sessions[sid] = copy.deepcopy(cls.DEFAULT_SESSION_STATE)
                cls._fallback_sessions[sid]["_initialized"] = True
            return cls._fallback_sessions[sid]

    @classmethod
    def init_session(cls, session_id: Optional[str] = None):
        if session_id:
            cls.set_session_id(session_id)
            
        with cls._acquire_lock():
            state = cls._get_state()
            if not state.get("_initialized", False):
                logger.info(f"[Session] [Init] 세션 상태 초기화 중... (ID: {cls.get_session_id()})")
                for key, value in cls.DEFAULT_SESSION_STATE.items():
                    if key not in state:
                        state[key] = copy.deepcopy(value)
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
            if "doc_pool" not in state:
                state["doc_pool"] = {}
            
            # [최적화] 문서 객체가 있으면 풀링 처리
            documents = kwargs.get("documents")
            if documents:
                doc_ids = []
                for doc in documents:
                    # 내용 및 출처 기반 해시 생성 (메타데이터 충돌 방지)
                    doc_key = f"{doc.page_content}_{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
                    content_hash = hashlib.sha256(doc_key.encode()).hexdigest()[:16]
                    if content_hash not in state["doc_pool"]:
                        state["doc_pool"][content_hash] = doc
                    doc_ids.append(content_hash)
                
                # 원본 documents 대신 ID 리스트 저장
                kwargs["doc_ids"] = doc_ids
                del kwargs["documents"]

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
            
            # [수정] 기존 로그 보존하고 분석 시작 알림 추가
            if "status_logs" not in state:
                state["status_logs"] = []
            state["status_logs"].append("--- 새 문서 분석 시작 ---")

    @classmethod
    def add_status_log(cls, msg: str):
        """작업 로그를 추가합니다. (최신 30개 보관)"""
        target = cls if not isinstance(cls, type) else None
        with ThreadSafeSessionManager._acquire_lock(instance=target):
            state = cls._get_state()
            if "status_logs" not in state:
                state["status_logs"] = []
            
            if state["status_logs"] and state["status_logs"][-1] == msg:
                return
                
            state["status_logs"].append(msg)
            # [수정] 히스토리 유지 개수 상향 (10 -> 30)
            if len(state["status_logs"]) > 30:
                state["status_logs"] = state["status_logs"][-30:]

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