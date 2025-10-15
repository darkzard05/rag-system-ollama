"Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일."

import time
import logging
import asyncio
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_mermaid import st_mermaid
import fitz  # PyMuPDF

from session import SessionManager
from model_loader import get_available_models
from config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    MSG_PREPARING_ANSWER,
    UI_CONTAINER_HEIGHT,
)


async def _stream_chat_response(qa_chain, user_input, chat_container) -> str:
    """
    LLM의 답변을 실시간으로 UI에 스트리밍하고 최종 답변 문자열을 반환합니다.
    """
    full_response = ""
    start_time = time.time()
    
    with chat_container, st.chat_message("assistant"):
        answer_container = st.empty()
        answer_container.markdown(MSG_PREPARING_ANSWER)

        try:
            async for event in qa_chain.astream_events(
                {"input": user_input}, version="v1"
            ):
                kind = event["event"]
                name = event.get("name", "")
                
                if kind == "on_chain_stream" and name == "generate_response":
                    chunk_data = event["data"].get("chunk")
                    
                    if isinstance(chunk_data, dict) and "response" in chunk_data:
                        full_response += chunk_data["response"]
                        answer_container.markdown(full_response + "▌")

        except Exception as e:
            error_msg = f"스트리밍 답변 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            answer_container.error(error_msg)
            return f"❌ {error_msg}"

    # --- 스트리밍 완료 후 최종 내용 정리 및 로그 출력 ---
    end_time = time.time()
    answer_container.markdown(full_response)
    
    total_duration = end_time - start_time
    perf_details = f"답변: {total_duration:.2f}초 ({len(full_response)}자)"
    logging.info(f"  [성능 상세] {perf_details}")
    
    return full_response


def render_sidebar(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback
):
    with st.sidebar:
        st.header("⚙️ 설정")
        st.file_uploader(
            "PDF 파일 업로드",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
        )
        with st.spinner("LLM 모델 목록을 불러오는 중..."):
            available_models = get_available_models()
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
                "LLM 모델 선택",
                available_models,
                index=current_model_index,
                key="model_selector",
                on_change=model_selector_callback,
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
            "임베딩 모델 선택",
            AVAILABLE_EMBEDDING_MODELS,
            index=current_embedding_model_index,
            key="embedding_model_selector",
            on_change=embedding_selector_callback,
        )
        st.divider()
        st.header("📊 시스템 상태")
        status_container = st.container()
        return status_container


def render_pdf_viewer():
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
        pdf_viewer(
            input=pdf_bytes,
            height=UI_CONTAINER_HEIGHT,
            pages_to_render=[st.session_state.current_page],
        )
        nav_cols = st.columns([1, 2, 1])
        with nav_cols[0]:
            st.button(
                "← 이전",
                on_click=go_to_previous_page,
                use_container_width=True,
                disabled=(st.session_state.current_page <= 1),
            )
        with nav_cols[1]:
            def sync_slider_and_input():
                st.session_state.current_page = st.session_state.current_page_slider
            st.slider(
                "페이지 이동",
                min_value=1,
                max_value=total_pages,
                key="current_page_slider",
                label_visibility="collapsed",
                value=st.session_state.current_page,
                on_change=sync_slider_and_input,
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
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    messages = SessionManager.get_messages()
    for message in messages:
        with chat_container, st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if user_input := st.chat_input(
        "PDF 내용에 대해 질문해보세요.", disabled=not SessionManager.is_ready_for_chat()
    ):
        SessionManager.add_message("user", user_input)
        st.rerun()

    if messages and messages[-1]["role"] == "user":
        last_user_input = messages[-1]["content"]
        qa_chain = SessionManager.get("qa_chain")
        
        if qa_chain:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            final_answer = loop.run_until_complete(
                _stream_chat_response(qa_chain, last_user_input, chat_container)
            )
            
            if final_answer:
                SessionManager.add_message("assistant", content=final_answer)
                st.rerun()
        else:
            st.error("QA 시스템이 준비되지 않았습니다. PDF를 먼저 처리해주세요.")

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

    if error_msg := SessionManager.get("pdf_processing_error"):
        st.error(f"오류가 발생했습니다: {error_msg}")
        if st.button("재시도"):
            SessionManager.reset_all_state()
            st.rerun()

def render_workflow_tab_content():
    """워크플로우 탭에 들어갈 콘텐츠를 렌더링합니다."""
    qa_chain = SessionManager.get("qa_chain")
    
    if not qa_chain:
        st.info("워크플로우를 보려면 먼저 사이드바에서 PDF를 처리해야 합니다.")
        return

    # 1. LangGraph에서 Mermaid 구문 추출
    try:
        mermaid_syntax = qa_chain.get_graph().draw_mermaid()
    except Exception as e:
        st.error(f"그래프를 생성하는 중 오류가 발생했습니다: {e}")
        return

    # 2. Mermaid 그래프 렌더링
    st.subheader("워크플로우 다이어그램")
    st_mermaid(mermaid_syntax, height="350px") # 컬럼에 맞게 높이 살짝 조절

    # 3. 각 노드에 대한 상세 설명 추가
    st.subheader("각 단계 설명")
    
    node_descriptions = {
        "retrieve": "**1. 문서 검색 (Retrieve):** 사용자의 질문과 가장 관련성이 높은 문서 조각들을 찾아냅니다.",
        "format_context": "**2. 컨텍스트 구성 (Format Context):** 검색된 문서 조각들을 LLM이 이해하기 쉬운 형식으로 정리합니다.",
        "generate_response": "**3. 답변 생성 (Generate Response):** 정리된 컨텍스트와 질문을 기반으로 최종 답변을 생성합니다."
    }

    graph_nodes = qa_chain.get_graph().nodes
    for node_name in graph_nodes:
        if node_name in node_descriptions:
            st.markdown(node_descriptions[node_name])
        elif node_name != "__end__":
            st.markdown(f"- **{node_name}**: 커스텀 노드")

def render_left_column_with_tabs():
    """
    왼쪽 컬럼에 '채팅'과 '워크플로우' 탭을 생성하고,
    각 탭에 맞는 콘텐츠를 렌더링합니다.
    """
    # 1. 탭 생성
    tab_chat, tab_workflow = st.tabs(["💬 채팅", "📊 워크플로우"])

    # 2. '채팅' 탭 콘텐츠 구성
    with tab_chat:
        # 기존 채팅 렌더링 함수를 호출합니다.
        # (이후 단계에서 이 함수를 약간 수정할 것입니다.)
        render_chat_column()

    # 3. '워크플로우' 탭 콘텐츠 구성
    with tab_workflow:
        # 그래프 뷰 렌더링 함수를 호출합니다.
        # (이후 단계에서 이 함수를 약간 수정할 것입니다.)
        render_workflow_tab_content() # 새 이름으로 변경