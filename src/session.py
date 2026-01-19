"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
이 클래스는 st.session_state에 대한 직접적인 Getter/Setter 역할에 집중합니다.
"""

import logging
import streamlit as st
from typing import List, Any, Dict


logger = logging.getLogger(__name__)

# ✅ 메모리 누수 방지: 최대 메시지 히스토리
MAX_MESSAGE_HISTORY = 1000


class SessionManager:
    """
    세션 상태를 관리하는 클래스 (순수 Getter/Setter 역할)
    """
    DEFAULT_SESSION_STATE: Dict[str, Any] = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_selected_embedding_model": None,
        "last_pdf_name": None,  # PDF 뷰어의 현재 파일 이름 추적
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_file_path": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "embedder": None,
        "is_generating_answer": False,
        "pdf_interaction_blocked": False,  # 답변 생성 중 PDF 상호작용 차단 플래그
        "is_first_run": True,
        "needs_rag_rebuild": False,
        "needs_qa_chain_update": False,
        "new_file_uploaded": False,
        "show_graph": False,
    }

    @classmethod
    def init_session(cls):
        """
        세션 상태 초기화 - 한 번만 실행되어야 함
        """
        if not st.session_state.get("_initialized", False):
            logger.info("세션 상태 초기화 중...")
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state._initialized = True
            logger.info("세션 상태 초기화 완료.")

    @classmethod
    def reset_for_new_file(cls):
        """
        새 파일 업로드 시 RAG 관련 상태를 안전하게 리셋합니다.
        """
        logger.info("새 파일 업로드 감지. RAG 상태 리셋.")

        keys_to_reset = [
            "pdf_processed",
            "pdf_processing_error",
            "qa_chain",
            "vector_store",
        ]

        for key in keys_to_reset:
            if key in st.session_state:
                st.session_state[key] = None

        st.session_state.pdf_processed = False
        st.session_state.needs_rag_rebuild = True
        st.session_state._chat_ready_needs_refresh = True

    @classmethod
    def add_message(cls, role: str, content: str):
        """
        메시지를 세션에 추가합니다.

        Args:
            role (str): 메시지의 역할 ("user" 또는 "assistant").
            content (str): 메시지 내용.
        """
        if "messages" not in st.session_state or not isinstance(
            st.session_state.messages, list
        ):
            st.session_state.messages = []

        message = {"role": role, "content": content}
        st.session_state.messages.append(message)
        
        # ✅ 메모리 누수 방지: 최대 히스토리 초과 시 오래된 메시지 제거
        if len(st.session_state.messages) > MAX_MESSAGE_HISTORY:
            removed_count = len(st.session_state.messages) - MAX_MESSAGE_HISTORY
            st.session_state.messages = st.session_state.messages[removed_count:]
            logger.info(f"메시지 제한 초과로 {removed_count}개 이전 메시지 삭제.")

    @staticmethod
    def is_ready_for_chat() -> bool:
        """
        채팅 준비 상태를 확인합니다 (캐시됨).

        Returns:
            bool: PDF 처리 완료, 에러 없음, QA 체인 준비 시 True.
        """
        if not st.session_state.get("_chat_ready_needs_refresh", True):
            return st.session_state.get("_cached_chat_ready", False)

        result = (
            st.session_state.get("pdf_processed", False)
            and not st.session_state.get("pdf_processing_error")
            and st.session_state.get("qa_chain") is not None
        )

        st.session_state._cached_chat_ready = result
        st.session_state._chat_ready_needs_refresh = False
        return result

    # --- Getters ---
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        세션 상태에서 값을 가져옵니다.
        """
        return st.session_state.get(key, default)

    @classmethod
    def get_messages(cls) -> List[Dict[str, str]]:
        """
        메시지 목록을 가져옵니다.
        """
        return st.session_state.get("messages", [])

    # --- Setters ---
    @classmethod
    def set(cls, key: str, value: Any):
        """
        세션 상태에 값을 설정합니다.
        """
        st.session_state[key] = value

    @classmethod
    def reset_all_state(cls):
        """
        세션의 모든 상태를 기본값으로 리셋합니다.
        """
        logger.info("모든 세션 상태 리셋.")

        # 위젯 키(보통 UI 컴포넌트의 key)를 제외하고 데이터만 리셋하는 것이 안전함
        # 혹은 DEFAULT_SESSION_STATE에 정의된 키만 초기화
        for key, value in cls.DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value
        
        # 초기화 플래그 유지
        st.session_state._initialized = True
