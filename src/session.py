"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
"""
import logging
import streamlit as st

class SessionManager:
    """세션 상태를 관리하는 클래스"""
    DEFAULT_SESSION_STATE = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_model_change_message": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_is_processing": False,
        "temp_pdf_path": None,
        "processed_document_splits": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
    }

    PRESERVE_ON_NEW_FILE_KEYS = [
        "_initialized",
        "last_selected_model",
        "last_model_change_message"
    ]

    @classmethod
    def init_session(cls):
        """세션 상태 초기화 - 한 번만 실행되어야 함"""
        if not st.session_state.get("_initialized", False):
            logging.info("세션 상태 초기화 중...")
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state._initialized = True

    @classmethod
    def reset_session_state(cls, keys=None):
        """지정된 키들의 세션 상태를 기본값으로 리셋"""
        keys_to_reset = keys if keys is not None else cls.DEFAULT_SESSION_STATE.keys()
        for key in keys_to_reset:
            if key in cls.DEFAULT_SESSION_STATE:
                st.session_state[key] = cls.DEFAULT_SESSION_STATE[key]

    @classmethod
    def reset_for_new_file(cls, uploaded_file):
        """새 파일 업로드시 세션 상태를 리셋합니다."""
        logging.info(f"새 파일 '{uploaded_file.name}' 업로드로 인한 세션 상태 리셋 중...")
        preserved_states = {
            key: st.session_state.get(key) for key in cls.PRESERVE_ON_NEW_FILE_KEYS
            if key in st.session_state
        }
        for key, value in cls.DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value
        for key, value in preserved_states.items():
            st.session_state[key] = value
        st.session_state.last_uploaded_file_name = uploaded_file.name
        if preserved_states.get("last_model_change_message"):
            cls.add_message("assistant", preserved_states["last_model_change_message"])

    @classmethod
    def add_message(cls, role: str, content: str):
        """메시지 추가"""
        if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})

    @staticmethod
    def is_ready_for_chat():
        """채팅 준비 상태 확인"""
        return (st.session_state.get("pdf_processed") and
                not st.session_state.get("pdf_processing_error") and
                st.session_state.get("qa_chain") is not None)

    @classmethod
    def update_model(cls, new_model: str):
        """모델 변경 시 관련 상태 리셋"""
        old_model = st.session_state.get("last_selected_model", "N/A")
        cls.reset_session_state(["llm", "qa_chain"])
        st.session_state.last_selected_model = new_model
        msg = f"🔄 모델을 '{new_model}'로 변경했습니다."
        cls.add_message("assistant", msg)
        st.session_state.last_model_change_message = msg
        return old_model

    @classmethod
    def set_error_state(cls, error_message: str):
        """에러 상태 설정"""
        st.session_state.pdf_processing_error = error_message
        cls.add_message("assistant", f"❌ {error_message}")
