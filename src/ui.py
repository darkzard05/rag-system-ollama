"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
"""

import time
import logging
import asyncio
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF

from session import SessionManager
from model_loader import get_available_models
from utils import sync_run
from config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    MSG_PREPARING_ANSWER,
    UI_CONTAINER_HEIGHT,
    MSG_STREAMING_ERROR,
    MSG_SIDEBAR_TITLE,
    MSG_PDF_UPLOADER_LABEL,
    MSG_LOADING_MODELS,
    MSG_MODEL_SELECTOR_LABEL,
    MSG_EMBEDDING_SELECTOR_LABEL,
    MSG_SYSTEM_STATUS_TITLE,
    MSG_PDF_VIEWER_TITLE,
    MSG_PDF_VIEWER_NO_FILE,
    MSG_PDF_VIEWER_PREV_BUTTON,
    MSG_PDF_VIEWER_PAGE_SLIDER,
    MSG_PDF_VIEWER_NEXT_BUTTON,
    MSG_PDF_VIEWER_ERROR,
    MSG_CHAT_TITLE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_CHAT_GUIDE,
    MSG_GENERIC_ERROR,
    MSG_RETRY_BUTTON,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)


logger = logging.getLogger(__name__)


async def _stream_chat_response(qa_chain, user_input, chat_container) -> str:
    """
    LLM의 답변을 실시간으로 UI에 스트리밍하고 최종 답변 문자열을 반환합니다.

    Args:
        qa_chain: RAG QA 체인 객체.
        user_input (str): 사용자가 입력한 질문.
        chat_container: Streamlit 채팅 메시지 컨테이너.

    Returns:
        str: 최종 생성된 답변 문자열.
    """
    full_response = ""
    start_time = time.time()

    current_llm = SessionManager.get("llm")

    if not current_llm:
        return "❌ 오류: 로드된 LLM 모델이 없습니다. 모델을 다시 선택해주세요."

    run_config = {
        "configurable": {
            "llm": current_llm
        }
    }

    SessionManager.set("is_generating_answer", True)
    SessionManager.set("pdf_interaction_blocked", True)

    try:
        with chat_container, st.chat_message("assistant"):
            answer_container = st.empty()
            answer_container.markdown(MSG_PREPARING_ANSWER)
            
            last_update_time = time.time()
            update_interval = 0.05  # 0.05초 간격으로 UI 업데이트 (부하 감소)

            try:
                async for event in qa_chain.astream_events(
                    {"input": user_input},
                    config=run_config,
                    version="v1"
                ):
                    kind = event["event"]
                    name = event.get("name", "")

                    if kind == "on_chain_stream" and name == "generate_response":
                        chunk_data = event["data"].get("chunk")

                        if isinstance(chunk_data, dict) and "response" in chunk_data:
                            full_response += chunk_data["response"]
                            
                            # 성능 최적화: 일정 시간 간격으로만 UI 업데이트
                            current_time = time.time()
                            if current_time - last_update_time > update_interval:
                                answer_container.markdown(full_response + "▌")
                                last_update_time = current_time
                                # 이벤트 루프에 제어권을 양보하여 UI 패킷 전송을 도움
                                await asyncio.sleep(0)

            except Exception as e:
                error_msg = MSG_STREAMING_ERROR.format(e=str(e))
                logger.error(f"Streaming error: {error_msg}", exc_info=True)
                answer_container.error(error_msg)
                return f"❌ {error_msg}"

        end_time = time.time()
        answer_container.markdown(full_response)

        total_duration = end_time - start_time
        perf_details = f"Response generated in {total_duration:.2f} seconds ({len(full_response)} chars)."
        logger.info(f"  [Performance] {perf_details}")

        return full_response

    finally:
        SessionManager.set("is_generating_answer", False)
        SessionManager.set("pdf_interaction_blocked", False)


def render_sidebar(
    file_uploader_callback, model_selector_callback, embedding_selector_callback
):
    """
    사이드바를 렌더링하고 콜백 함수를 설정합니다.

    Args:
        file_uploader_callback: 파일 업로더 변경 시 호출되는 콜백.
        model_selector_callback: 모델 선택 변경 시 호출되는 콜백.
        embedding_selector_callback: 임베딩 모델 선택 변경 시 호출되는 콜백.

    Returns:
        Container: 상태 메시지를 표시할 컨테이너.
    """
    with st.sidebar:
        st.header(MSG_SIDEBAR_TITLE)
        st.file_uploader(
            MSG_PDF_UPLOADER_LABEL,
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
        )

        with st.spinner(MSG_LOADING_MODELS):
            available_models = get_available_models()
            is_ollama_error = (
                available_models and available_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING
            )

        if is_ollama_error:
            actual_models = []
        else:
            actual_models = [m for m in available_models if "---" not in m]

        last_model = SessionManager.get("last_selected_model")
        if not last_model or last_model not in actual_models:
            if actual_models:
                last_model = actual_models[0]
                SessionManager.set("last_selected_model", last_model)
            else:
                last_model = OLLAMA_MODEL_NAME

        current_model_index = (
            available_models.index(last_model)
            if last_model in available_models
            else 0
        )

        st.selectbox(
            MSG_MODEL_SELECTOR_LABEL,
            available_models,
            index=current_model_index,
            key="model_selector",
            on_change=model_selector_callback,
            disabled=is_ollama_error,
        )

        last_embedding_model = (
            SessionManager.get("last_selected_embedding_model")
            or AVAILABLE_EMBEDDING_MODELS[0]
        )
        current_embedding_model_index = (
            AVAILABLE_EMBEDDING_MODELS.index(last_embedding_model)
            if last_embedding_model in AVAILABLE_EMBEDDING_MODELS
            else 0
        )
        st.selectbox(
            MSG_EMBEDDING_SELECTOR_LABEL,
            AVAILABLE_EMBEDDING_MODELS,
            index=current_embedding_model_index,
            key="embedding_model_selector",
            on_change=embedding_selector_callback,
        )
        st.header(MSG_SYSTEM_STATUS_TITLE)
        status_container = st.container()

        if is_ollama_error:
            status_container.error(available_models[0])

        return status_container


def render_pdf_viewer():
    """
    PDF 뷰어를 렌더링합니다.
    """
    _pdf_viewer_fragment()


def _pdf_viewer_fragment():
    """
    PDF 뷰어 렌더링 로직 (fragment로 분리).
    """
    st.subheader(MSG_PDF_VIEWER_TITLE)

    pdf_bytes = SessionManager.get("pdf_file_bytes")
    if not pdf_bytes:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return
    
    pdf_document = None
    try:
        # ✅ Context manager로 PDF 리소스 자동 정리
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_document)
        current_file_name = SessionManager.get("last_uploaded_file_name")
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if st.session_state.get("last_pdf_name") != current_file_name:
            st.session_state.current_page = 1
            st.session_state.last_pdf_name = current_file_name
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = 1

        def go_to_previous_page():
            # 답변 생성 중이면 무시
            if SessionManager.get("is_generating_answer", False):
                return
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1

        def go_to_next_page():
            # 답변 생성 중이면 무시
            if SessionManager.get("is_generating_answer", False):
                return
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1

        pdf_viewer(
            input=pdf_bytes,
            height=UI_CONTAINER_HEIGHT,
            pages_to_render=[st.session_state.current_page],
        )
        nav_cols = st.columns([1, 2, 1])
        with nav_cols[0]:
            st.button(
                MSG_PDF_VIEWER_PREV_BUTTON,
                on_click=go_to_previous_page,
                use_container_width=True,
                disabled=(st.session_state.current_page <= 1),
            )
        with nav_cols[1]:

            def sync_slider_and_input():
                # 답변 생성 중이면 슬라이더 조작 무시
                if SessionManager.get("is_generating_answer", False):
                    # 이전 값으로 복원
                    st.session_state.current_page_slider = st.session_state.current_page
                    return
                st.session_state.current_page = st.session_state.current_page_slider

            st.slider(
                MSG_PDF_VIEWER_PAGE_SLIDER,
                min_value=1,
                max_value=total_pages,
                key="current_page_slider",
                label_visibility="collapsed",
                value=st.session_state.current_page,
                on_change=sync_slider_and_input,
            )
        with nav_cols[2]:
            st.button(
                MSG_PDF_VIEWER_NEXT_BUTTON,
                on_click=go_to_next_page,
                use_container_width=True,
                disabled=(st.session_state.current_page >= total_pages),
            )
    except Exception as e:
        st.error(MSG_PDF_VIEWER_ERROR.format(e=e))
        logger.error("PDF viewer error", exc_info=True)
    finally:
        # ✅ PDF 리소스 명시적 정리 (메모리 누수 방지)
        if pdf_document is not None:
            pdf_document.close()


def render_chat_column():
    """
    채팅 컬럼을 렌더링하고 채팅 로직을 처리합니다.
    """
    _chat_fragment()


def _chat_fragment():
    """
    채팅 렌더링 로직 (fragment로 분리).
    """
    st.subheader(MSG_CHAT_TITLE)
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    messages = SessionManager.get_messages()
    for message in messages:
        with chat_container, st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input(
        MSG_CHAT_INPUT_PLACEHOLDER, 
        disabled=not SessionManager.is_ready_for_chat() or SessionManager.get("is_generating_answer")
    ):
        SessionManager.add_message("user", user_input)
        st.rerun()

    if messages and messages[-1]["role"] == "user":
        last_user_input = messages[-1]["content"]
        qa_chain = SessionManager.get("qa_chain")

        if qa_chain:
            final_answer = sync_run(
                _stream_chat_response(qa_chain, last_user_input, chat_container)
            )

            if final_answer:
                SessionManager.add_message("assistant", content=final_answer)
                st.rerun()
        else:
            st.error(MSG_CHAT_NO_QA_SYSTEM)

    if not SessionManager.get_messages():
        with chat_container:
            st.info(MSG_CHAT_WELCOME)
            st.markdown(MSG_CHAT_GUIDE)

    if error_msg := SessionManager.get("pdf_processing_error"):
        st.error(MSG_GENERIC_ERROR.format(error_msg=error_msg))
        if st.button(MSG_RETRY_BUTTON):
            SessionManager.reset_all_state()
            st.rerun()


def render_left_column():
    """
    왼쪽 컬럼에 채팅 UI를 렌더링합니다.
    """
    render_chat_column()