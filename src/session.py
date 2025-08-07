"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
이 클래스는 st.session_state에 대한 직접적인 Getter/Setter 역할에 집중합니다.
"""
import logging
import streamlit as st
from typing import List, Any, Dict, Optional

class SessionManager:
    """세션 상태를 관리하는 클래스 (순수 Getter/Setter 역할)"""
    DEFAULT_SESSION_STATE: Dict[str, Any] = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_selected_embedding_model": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_bytes": None,
        "processed_document_splits": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "embedder": None,
        "is_generating_answer": False,
    }

    @classmethod
    def init_session(cls):
        """세션 상태 초기화 - 한 번만 실행되어야 함"""
        if not st.session_state.get("_initialized", False):
            logging.info("세션 상태 초기화 중...")
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state._initialized = True
            logging.info("세션 상태 초기화 완료.")

    @classmethod
    def reset_all_state(cls):
        """모든 세션 상태를 기본값으로 리셋합니다."""
        logging.info("모든 세션 상태를 기본값으로 리셋합니다.")
        for key, value in cls.DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value
        # 초기화 플래그는 유지
        st.session_state._initialized = True

    @classmethod
    def add_message(cls, role: str, content: str):
        """메시지 추가"""
        if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})

    @staticmethod
    def is_ready_for_chat() -> bool:
        """채팅 준비 상태 확인"""
        return (st.session_state.get("pdf_processed", False) and
                not st.session_state.get("pdf_processing_error") and
                st.session_state.get("qa_chain") is not None)

    # --- Getters ---
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @classmethod
    def get_messages(cls) -> List[Dict[str, str]]:
        return st.session_state.get("messages", [])

    # --- Setters ---
    @classmethod
    def set(cls, key: str, value: Any):
        st.session_state[key] = value