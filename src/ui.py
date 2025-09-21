"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
"""

import logging
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF

from session import SessionManager
from model_loader import get_available_models
from config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    MSG_PREPARING_ANSWER,
    MSG_THINKING,
    MSG_NO_THOUGHT_PROCESS,
    UI_CONTAINER_HEIGHT,
)


def _process_chat_response(qa_chain, user_input, chat_container):
    """
    LangGraph RAG 체인에서 스트리밍된 응답을 실시간으로 처리하고 UI에 표시합니다.
    """
    with chat_container, st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(MSG_PREPARING_ANSWER)

        final_generation = ""
        try:
            # LangGraph는 각 노드가 끝날 때마다 상태를 스트리밍합니다.
            stream = qa_chain.stream({"input": user_input})

            for chunk in stream:
                # 스트림의 각 청크는 {'노드이름': {'상태키': 값}} 형태입니다.
                # 마지막 'generate' 노드의 출력을 찾습니다.
                if "generate" in chunk:
                    generation_output = chunk["generate"].get("generation", "")
                    if generation_output:
                        final_generation = generation_output
                        message_placeholder.markdown(final_generation + "▌")
            
            # 스트리밍 완료 후 최종 UI 업데이트
            message_placeholder.markdown(final_generation)
            SessionManager.add_message("assistant", final_generation)
            logging.info(f"LangGraph 스트리밍 완료. (응답 길이: {len(final_generation)}자)")

        except Exception as e:
            error_msg = f"답변 스트리밍 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            SessionManager.add_message("assistant", f"❌ {error_msg}")


def render_sidebar(
    file_uploader_callback, model_selector_callback, embedding_selector_callback
):
    """사이드바 UI를 렌더링하고 사용자 입력을 처리합니다."""
    with st.sidebar:
        st.header("⚙️ 설정")

        # --- 파일 업로더 ---
        st.file_uploader(
            "PDF 파일 업로드",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
        )

        # --- LLM 모델 선택 (동적 목록) ---
        with st.spinner("LLM 모델 목록을 불러오는 중..."):
            available_models = get_available_models()

            # 사용 가능한 실제 모델 목록 (구분선 제외)
            actual_models = [m for m in available_models if "---" not in m]

            last_model = SessionManager.get("last_selected_model")

            # 세션에 모델이 없거나, 있더라도 현재 사용 불가능한 모델일 경우 기본값 설정
            if not last_model or last_model not in actual_models:
                if actual_models:
                    last_model = actual_models[0]
                    SessionManager.set("last_selected_model", last_model)
                else:  # 사용 가능한 모델이 전혀 없을 경우
                    last_model = OLLAMA_MODEL_NAME

            current_model_index = (
                available_models.index(last_model)
                if last_model in available_models
                else 0
            )

            st.selectbox(
                "LLM 모델 선택",
                available_models,
                index=current_model_index,
                key="model_selector",
                on_change=model_selector_callback,
            )

        # --- 임베딩 모델 선택 ---
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
            "임베딩 모델 선택",
            AVAILABLE_EMBEDDING_MODELS,
            index=current_embedding_model_index,
            key="embedding_model_selector",
            on_change=embedding_selector_callback,
        )

        # --- 상태 메시지 표시 영역 ---
        st.header("📊 시스템 상태")
        status_container = st.container()
        return status_container


def render_pdf_viewer():
    """PDF 뷰어 컬럼을 렌더링합니다."""
    st.subheader("📄 PDF 미리보기")

    pdf_bytes = SessionManager.get("pdf_file_bytes")
    if not pdf_bytes:
        st.info("미리볼 PDF가 없습니다. 사이드바에서 파일을 업로드해주세요.")
        return

    try:
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
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1

        def go_to_next_page():
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1

        # 1. PDF 뷰어 먼저 표시
        pdf_viewer(
            input=pdf_bytes,
            height=UI_CONTAINER_HEIGHT,
            pages_to_render=[st.session_state.current_page],
        )

        # 2. 내비게이션 UI 개선
        # 컬럼 비율을 조정하여 페이지 입력과 텍스트를 분리
        nav_cols = st.columns([1, 2, 1])

        with nav_cols[0]:
            st.button(
                "← 이전",
                on_click=go_to_previous_page,
                use_container_width=True,
                disabled=(st.session_state.current_page <= 1),
            )

        with nav_cols[1]:
            # 슬라이더와 number_input 동기화를 위한 콜백 함수
            def sync_slider_and_input():
                st.session_state.current_page = st.session_state.current_page_slider

            st.slider(
                "페이지 이동",
                min_value=1,
                max_value=total_pages,
                key="current_page_slider",
                label_visibility="collapsed",
                value=st.session_state.current_page,
                on_change=sync_slider_and_input,  # on_change 콜백 사용
            )

        with nav_cols[2]:
            st.button(
                "다음 →",
                on_click=go_to_next_page,
                use_container_width=True,
                disabled=(st.session_state.current_page >= total_pages),
            )

    except Exception as e:
        st.error(f"PDF를 표시하는 중 오류가 발생했습니다: {e}")
        logging.error("PDF 뷰어 오류", exc_info=True)


def render_chat_column():
    """채팅 컬럼을 렌더링하고 채팅 로직을 처리합니다."""
    st.subheader("💬 채팅")

    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    # 기존 메시지 표시
    for message in SessionManager.get_messages():
        with chat_container, st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # 사용자 입력 처리
    if user_input := st.chat_input(
        "PDF 내용에 대해 질문해보세요.", disabled=not SessionManager.is_ready_for_chat()
    ):
        SessionManager.add_message("user", user_input)
        with chat_container, st.chat_message("user"):
            st.markdown(user_input)

        qa_chain = SessionManager.get("qa_chain")
        if qa_chain:
            _process_chat_response(qa_chain, user_input, chat_container)
        else:
            st.error("QA 시스템이 준비되지 않았습니다. PDF를 먼저 처리해주세요.")

    # 초기 안내 메시지
    if not SessionManager.get_messages():
        with chat_container:
            st.info(
                "**RAG-Chat에 오신 것을 환영합니다!**\n\n"
                "사이드바에서 PDF 파일을 업로드하여 문서 내용에 대한 대화를 시작해보세요."
            )

            st.markdown(
                """
                **💡 사용 가이드**
                - **PDF 업로드:** 좌측 사이드바에서 분석하고 싶은 PDF를 업로드하세요.
                - **모델 선택:** 로컬 `Ollama` 모델 또는 `Gemini` API 모델을 선택할 수 있습니다.
                - **질문하기:** 문서 처리가 완료되면, 내용에 대해 자유롭게 질문할 수 있습니다.
                - **PDF 뷰어:** 우측에서 원본 문서를 함께 보며 대화할 수 있습니다.
                
                **⚠️ 알아두실 점**
                - **답변의 정확성:** 답변은 업로드된 PDF 내용만을 기반으로 생성되며, 사실이 아닐 수 있습니다.
                - **개인정보:** Gemini 모델 사용 시, 질문 내용이 Google 서버로 전송될 수 있습니다.
                - **초기 로딩:** 임베딩 모델을 처음 사용하면 다운로드에 몇 분이 소요될 수 있습니다.
                """
            )

    # 에러 메시지 표시
    if error_msg := SessionManager.get("pdf_processing_error"):
        st.error(f"오류가 발생했습니다: {error_msg}")
        if st.button("재시도"):
            SessionManager.reset_all_state()
            st.rerun()
