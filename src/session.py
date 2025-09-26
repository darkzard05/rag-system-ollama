"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
이 클래스는 st.session_state에 대한 직접적인 Getter/Setter 역할에 집중합니다.
"""

import logging
import streamlit as st
from typing import List, Any, Dict


class SessionManager:
    """세션 상태를 관리하는 클래스 (순수 Getter/Setter 역할)"""

    DEFAULT_SESSION_STATE: Dict[str, Any] = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_selected_embedding_model": None,
        "last_pdf_name": None, # PDF 뷰어의 현재 파일 이름 추적
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_bytes": None,
        "processed_document_splits": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "embedder": None,
        "is_generating_answer": False,
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "show_graph": False,
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
    def reset_for_new_file(cls):
        """새 파일 업로드 시 RAG 관련 상태를 안전하게 리셋합니다."""
        logging.info("새 파일 업로드 감지. RAG 관련 상태를 리셋합니다.")

        # 리셋할 키 목록
        keys_to_reset = [
            "pdf_processed",
            "pdf_processing_error",
            "processed_document_splits",
            "qa_chain",
            "vector_store",
        ]

        # st.session_state에서 해당 키들을 순회하며 삭제
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

        # 상태 플래그 설정
        st.session_state.pdf_processed = False
        st.session_state.needs_rag_rebuild = True

    @classmethod
    def add_message(cls, role: str, content: str, thought: str = None):
        """
        메시지 추가. 'thought' 인자를 추가하여 생각 과정을 별도로 저장합니다.
        """
        if "messages" not in st.session_state or not isinstance(
            st.session_state.messages, list
        ):
            st.session_state.messages = []
        
        message = {"role": role, "content": content}
        # 💡 'thought' 내용이 있으면 메시지 객체에 추가
        if thought:
            message["thought"] = thought
            
        st.session_state.messages.append(message)

    @staticmethod
    def is_ready_for_chat() -> bool:
        """채팅 준비 상태 확인"""
        return (
            st.session_state.get("pdf_processed", False)
            and not st.session_state.get("pdf_processing_error")
            and st.session_state.get("qa_chain") is not None
        )

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

    @classmethod
    def reset_all_state(cls):
        """세션의 모든 상태를 기본값으로 리셋합니다."""
        logging.info("세션의 모든 상태를 리셋합니다.")
        # st.session_state.clear() # clear()는 모든 것을 지우므로 콜백과 위젯 상태에 문제를 일으킬 수 있음
        for key in list(st.session_state.keys()):
            if key != "_initialized":  # 초기화 플래그는 유지
                del st.session_state[key]
        cls.init_session()
