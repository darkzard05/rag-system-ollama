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
        "last_selected_embedding_model": None, # Added
        "pdf_processed": False,
        "pdf_processing_error": None,
        "temp_pdf_path": None,
        "processed_document_splits": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "embedder": None, # Added
        "resolution_boost": 1, # Added
        "pdf_width": 1000, # Added
        "pdf_height": 1000, # Added
    }

    PRESERVE_ON_NEW_FILE_KEYS = [
        "_initialized",
        "last_selected_model",
        "last_model_change_message",
        "last_selected_embedding_model", # Added
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
            logging.info("세션 상태 초기화 완료.")

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

        # Preserve states that should carry over
        preserved_model = cls.get_last_selected_model()
        preserved_embedding_model = cls.get_last_selected_embedding_model()
        preserved_model_change_message = st.session_state.get("last_model_change_message") # Get directly before reset

        # Reset all session state to default
        for key, value in cls.DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value

        # Restore preserved states
        if preserved_model:
            cls.set_last_selected_model(preserved_model)
        if preserved_embedding_model:
            cls.set_last_selected_embedding_model(preserved_embedding_model)
        
        # Set new file name
        cls.set_last_uploaded_file_name(uploaded_file.name)

        # Re-add the model change message if it existed
        if preserved_model_change_message:
            cls.add_message("assistant", preserved_model_change_message)
        

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

    @classmethod
    def get_messages(cls):
        return st.session_state.get("messages", [])

    @classmethod
    def get_last_selected_model(cls):
        return st.session_state.get("last_selected_model")

    @classmethod
    def set_last_selected_model(cls, model_name: str):
        st.session_state.last_selected_model = model_name

    @classmethod
    def get_last_uploaded_file_name(cls):
        return st.session_state.get("last_uploaded_file_name")

    @classmethod
    def set_last_uploaded_file_name(cls, file_name: str):
        st.session_state.last_uploaded_file_name = file_name

    @classmethod
    def get_pdf_processed(cls):
        return st.session_state.get("pdf_processed", False)

    @classmethod
    def set_pdf_processed(cls, value: bool):
        st.session_state.pdf_processed = value

    @classmethod
    def get_pdf_processing_error(cls):
        return st.session_state.get("pdf_processing_error")

    @classmethod
    def get_temp_pdf_path(cls):
        return st.session_state.get("temp_pdf_path")

    @classmethod
    def set_temp_pdf_path(cls, path: str):
        st.session_state.temp_pdf_path = path

    @classmethod
    def get_processed_document_splits(cls):
        return st.session_state.get("processed_document_splits")

    @classmethod
    def set_processed_document_splits(cls, splits):
        st.session_state.processed_document_splits = splits

    @classmethod
    def get_qa_chain(cls):
        return st.session_state.get("qa_chain")

    @classmethod
    def set_qa_chain(cls, chain):
        st.session_state.qa_chain = chain

    @classmethod
    def get_vector_store(cls):
        return st.session_state.get("vector_store")

    @classmethod
    def set_vector_store(cls, store):
        st.session_state.vector_store = store

    @classmethod
    def get_llm(cls):
        return st.session_state.get("llm")

    @classmethod
    def set_llm(cls, llm):
        st.session_state.llm = llm

    @classmethod
    def get_embedder(cls):
        return st.session_state.get("embedder")

    @classmethod
    def set_embedder(cls, embedder):
        st.session_state.embedder = embedder

    @classmethod
    def get_last_selected_embedding_model(cls):
        return st.session_state.get("last_selected_embedding_model")

    @classmethod
    def set_last_selected_embedding_model(cls, model_name: str):
        st.session_state.last_selected_embedding_model = model_name

    @classmethod
    def get_resolution_boost(cls):
        return st.session_state.get("resolution_boost", cls.DEFAULT_SESSION_STATE["resolution_boost"])

    @classmethod
    def set_resolution_boost(cls, value: int):
        st.session_state.resolution_boost = value

    @classmethod
    def get_pdf_width(cls):
        return st.session_state.get("pdf_width", cls.DEFAULT_SESSION_STATE["pdf_width"])

    @classmethod
    def set_pdf_width(cls, value: int):
        st.session_state.pdf_width = value

    @classmethod
    def get_pdf_height(cls):
        return st.session_state.get("pdf_height", cls.DEFAULT_SESSION_STATE["pdf_height"])

    @classmethod
    def set_pdf_height(cls, value: int):
        st.session_state.pdf_height = value
