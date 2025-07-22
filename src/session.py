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
        "last_selected_embedding_model": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_bytes": None, # temp_pdf_path 대신 사용
        "processed_document_splits": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "embedder": None,
        "pdf_width": 1000,
        "pdf_height": 1000,
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
    def reset_for_new_file(cls, file_name: str, file_bytes: bytes):
        """새 파일 업로드시 세션 상태를 리셋하고 새 파일 정보를 즉시 설정합니다."""
        logging.info(f"새 파일 '{file_name}' 업로드로 인한 세션 상태 리셋 중...")

        preserved_model = cls.get_last_selected_model()
        preserved_embedding_model = cls.get_last_selected_embedding_model()
        
        # 모든 세션 상태를 기본값으로 리셋
        for key, value in cls.DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value

        # 보존해야 할 값들 복원
        cls.set_last_selected_model(preserved_model)
        cls.set_last_selected_embedding_model(preserved_embedding_model)
        
        # 새 파일 정보 즉시 설정
        cls.set_last_uploaded_file_name(file_name)
        cls.set_pdf_file_bytes(file_bytes)
        
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
        """모델 변경 시 관련 상태 리셋. 초기 설정 시에는 메시지를 추가하지 않음."""
        last_model = cls.get_last_selected_model()
        st.session_state.last_selected_model = new_model

        # Only add a message if the model was already set to something else.
        if last_model is not None:
            msg = f"🔄 모델을 '{new_model}'로 변경했습니다."
            cls.add_message("assistant", msg)

    @classmethod
    def set_error_state(cls, error_message: str):
        """에러 상태 설정"""
        st.session_state.pdf_processing_error = error_message
        cls.add_message("assistant", f"❌ {error_message}")

    # --- Getters and Setters ---
    @classmethod
    def get_messages(cls): return st.session_state.get("messages", [])
    @classmethod
    def get_last_selected_model(cls): return st.session_state.get("last_selected_model")
    @classmethod
    def set_last_selected_model(cls, model_name: str): st.session_state.last_selected_model = model_name
    @classmethod
    def get_last_uploaded_file_name(cls): return st.session_state.get("last_uploaded_file_name")
    @classmethod
    def set_last_uploaded_file_name(cls, file_name: str): st.session_state.last_uploaded_file_name = file_name
    @classmethod
    def get_pdf_processed(cls): return st.session_state.get("pdf_processed", False)
    @classmethod
    def set_pdf_processed(cls, value: bool): st.session_state.pdf_processed = value
    @classmethod
    def get_pdf_processing_error(cls): return st.session_state.get("pdf_processing_error")
    @classmethod
    def get_pdf_file_bytes(cls): return st.session_state.get("pdf_file_bytes")
    @classmethod
    def set_pdf_file_bytes(cls, file_bytes: bytes): st.session_state.pdf_file_bytes = file_bytes
    @classmethod
    def get_processed_document_splits(cls): return st.session_state.get("processed_document_splits")
    @classmethod
    def set_processed_document_splits(cls, splits): st.session_state.processed_document_splits = splits
    @classmethod
    def get_qa_chain(cls): return st.session_state.get("qa_chain")
    @classmethod
    def set_qa_chain(cls, chain): st.session_state.qa_chain = chain
    @classmethod
    def get_vector_store(cls): return st.session_state.get("vector_store")
    @classmethod
    def set_vector_store(cls, store): st.session_state.vector_store = store
    @classmethod
    def get_llm(cls): return st.session_state.get("llm")
    @classmethod
    def set_llm(cls, llm): st.session_state.llm = llm
    @classmethod
    def get_embedder(cls): return st.session_state.get("embedder")
    @classmethod
    def set_embedder(cls, embedder): st.session_state.embedder = embedder
    @classmethod
    def get_last_selected_embedding_model(cls): return st.session_state.get("last_selected_embedding_model")
    @classmethod
    def set_last_selected_embedding_model(cls, model_name: str): st.session_state.last_selected_embedding_model = model_name
    @classmethod
    def get_pdf_width(cls): return st.session_state.get("pdf_width", cls.DEFAULT_SESSION_STATE["pdf_width"])
    @classmethod
    def set_pdf_width(cls, value: int): st.session_state.pdf_width = value
    @classmethod
    def get_pdf_height(cls): return st.session_state.get("pdf_height", cls.DEFAULT_SESSION_STATE["pdf_height"])
    @classmethod
    def set_pdf_height(cls, value: int): st.session_state.pdf_height = value
