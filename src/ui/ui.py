"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

import time
import logging
import os
import re
from contextlib import aclosing
from typing import Callable, Optional

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF

from common.exceptions import (
    PDFProcessingError,
    EmptyPDFError,
    InsufficientChunksError,
    VectorStoreError,
    LLMInferenceError,
    EmbeddingModelError,
)
from core.session import SessionManager
from core.model_loader import get_available_models
from common.utils import sync_run, apply_tooltips_to_response
from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    UI_CONTAINER_HEIGHT,
    MSG_SIDEBAR_TITLE,
    MSG_PDF_UPLOADER_LABEL,
    MSG_MODEL_SELECTOR_LABEL,
    MSG_EMBEDDING_SELECTOR_LABEL,
    MSG_SYSTEM_STATUS_TITLE,
    MSG_PDF_VIEWER_TITLE,
    MSG_PDF_VIEWER_NO_FILE,
    MSG_PDF_VIEWER_PREV_BUTTON,
    MSG_PDF_VIEWER_NEXT_BUTTON,
    MSG_PDF_VIEWER_ERROR,
    MSG_CHAT_TITLE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    MSG_PREPARING_ANSWER,
)

logger = logging.getLogger(__name__)


def _render_status_box(container):
    """시스템 상태 로그 박스를 지정된 컨테이너에 실시간으로 렌더링합니다."""
    if container is None:
        return
        
    status_logs = SessionManager.get("status_logs", [])
    
    # [스타일링: 하단 밀착형 롤링 로그]
    log_html = """
    <style>
    .status-container {
        font-family: 'Source Code Pro', monospace;
        padding: 15px !important;
        text-align: left !important;
        width: 100%;
        background-color: transparent !important;
        border: none !important;
        margin-top: -20px !important;
    }
    .status-line {
        line-height: 1.6;
        margin: 0px !important;
        padding: 0px !important;
        text-align: left !important;
    }
    .status-current { color: #0068c9; font-weight: bold; font-size: 0.85em; }
    .status-history { color: #888; font-size: 0.8em; }
    </style>
    """
    
    display_logs = status_logs[-4:] if status_logs else []
    padded_logs = [""] * (4 - len(display_logs)) + display_logs

    log_content = ""
    for i, log in enumerate(padded_logs):
        import re
        clean_log = re.sub(r'[^\x00-\x7F가-힣\s]', '', log).strip()
        if clean_log == "" and log != "": clean_log = log.strip()

        if clean_log == "":
            log_content += "<div class='status-line status-history'>&nbsp;</div>"
        elif i == 3:
            log_content += f"<div class='status-line status-current'>&gt; {clean_log}</div>"
        else:
            log_content += f"<div class='status-line status-history'>- {clean_log}</div>"
    
    full_html = f"{log_html}<div class='status-container'>{log_content}</div>"
    container.markdown(full_html, unsafe_allow_html=True)


async def _stream_chat_response(rag_engine, user_query: str, chat_container) -> str:
    """
    RAG 엔진의 이벤트를 수신하여 사고 과정과 답변을 실시간으로 렌더링합니다.
    """
    state = {
        "full_response": "",
        "full_thought": "",
        "retrieved_docs": [],
        "start_time": time.time()
    }
    
    current_llm = SessionManager.get("llm")
    if not current_llm:
        return "❌ 오류: LLM 모델이 로드되지 않았습니다."
        
    status_placeholder = SessionManager.get("status_placeholder")
    run_config = {"configurable": {"llm": current_llm}}
    SessionManager.set("is_generating_answer", True)

    try:
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                # 1. UI 컴포넌트 초기화
                thought_container = st.empty()
                with thought_container:
                    thought_expander = st.expander("🧠 사고 과정 (Thinking...)", expanded=True)
                    thought_display = thought_expander.empty()
                
                answer_display = st.empty()
                answer_display.markdown(f"⌛ {MSG_PREPARING_ANSWER}")
                
                # 2. 스트리밍 이벤트 처리 루프
                async with aclosing(rag_engine.astream_events(
                    {"input": user_query}, config=run_config, version="v2"
                )) as event_stream:
                    async for event in event_stream:
                        event_kind = event["event"]
                        event_name = event.get("name", "Unknown")
                        event_data = event.get("data", {})
                        
                        # 상태 박스 동기화
                        if event_kind in ["on_chain_start", "on_chain_end"]:
                            _render_status_box(status_placeholder)
                        
                        # [Integrity Protocol] 커스텀 응답 청크 이벤트 처리
                        if event_kind == "on_custom_event" and event_name == "response_chunk":
                            chunk_content = event_data.get("chunk")
                            chunk_thought = event_data.get("thought")

                            # 사고 과정 렌더링
                            if chunk_thought:
                                state["full_thought"] += chunk_thought
                                thought_display.markdown(state["full_thought"] + "▌")
                            
                            # 답변 본문 렌더링
                            if chunk_content:
                                state["full_response"] += chunk_content
                                answer_display.markdown(state["full_response"] + "▌", unsafe_allow_html=True)
                            
                        # 문서 검색 결과 캡처
                        elif event_kind == "on_chain_end":
                            if event_name == "retrieve":
                                output = event_data.get("output", {})
                                if "documents" in output:
                                    state["retrieved_docs"] = output["documents"]
                            
                            elif event_name == "generate_response":
                                output = event_data.get("output", {})
                                if isinstance(output, dict):
                                    # 최종 결과 확정 (유실 방지용 폴백)
                                    if "documents" in output and not state["retrieved_docs"]:
                                        state["retrieved_docs"] = output["documents"]
                                    if "response" in output and len(output["response"]) > len(state["full_response"]):
                                        state["full_response"] = output["response"]
                
                # 3. 답변 완료 후 UI 정리
                _finalize_ui_rendering(thought_container, answer_display, state)
                
        return state["full_response"]

    except Exception as e:
        logger.error(f"UI 스트리밍 오류: {e}", exc_info=True)
        return f"❌ 오류 발생: {str(e)}"
    finally:
        SessionManager.set("is_generating_answer", False)
        _render_status_box(status_placeholder)

def _finalize_ui_rendering(thought_container, answer_display, state):
    """답변 생성이 끝난 후 UI를 최종 상태로 정리합니다."""
    # 사고 과정 정리
    if state["full_thought"]:
        with thought_container:
            label = f"🧠 사고 완료 ({len(state['full_thought'].split())} tokens)"
            with st.expander(label, expanded=False):
                st.markdown(state["full_thought"])
    else:
        thought_container.empty()

    # 답변 본문 최종 렌더링 (툴팁 및 하이라이트 적용)
    if state["full_response"]:
        if state["retrieved_docs"]:
            final_html = apply_tooltips_to_response(state["full_response"], state["retrieved_docs"])
            answer_display.markdown(final_html, unsafe_allow_html=True)
        else:
            answer_display.markdown(state["full_response"], unsafe_allow_html=True)
    else:
        answer_display.error("⚠️ 답변이 생성되지 않았습니다.")


def render_sidebar(
    file_uploader_callback: Callable,
    model_selector_callback: Callable,
    embedding_selector_callback: Callable
):
    with st.sidebar:
        st.header(MSG_SIDEBAR_TITLE)
        is_generating = SessionManager.get("is_generating_answer")
        
        # --- 1. 문서 불러오기 ---
        with st.expander("📄 문서 불러오기", expanded=True):
            st.file_uploader(
                "PDF 파일 선택", 
                type="pdf", 
                key="pdf_uploader", 
                on_change=file_uploader_callback,
                disabled=is_generating,
                label_visibility="collapsed" # 중복 라벨 제거
            )

        # --- 2. 모델 설정 ---
        with st.expander("⚙️ 모델 설정", expanded=True):
            available_models = get_available_models()
            is_ollama_error = bool(available_models) and available_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING
            actual_models = [] if is_ollama_error else [m for m in available_models if "---" not in m]
            
            last_model = SessionManager.get("last_selected_model")
            if not last_model or (actual_models and last_model not in actual_models):
                last_model = actual_models[0] if actual_models else OLLAMA_MODEL_NAME
                SessionManager.set("last_selected_model", last_model)

            try: idx = available_models.index(last_model)
            except ValueError: idx = 0

            st.selectbox(MSG_MODEL_SELECTOR_LABEL, available_models, index=idx, key="model_selector", on_change=model_selector_callback, disabled=(is_ollama_error or is_generating))

            last_emb = SessionManager.get("last_selected_embedding_model") or AVAILABLE_EMBEDDING_MODELS[0]
            try: emb_idx = AVAILABLE_EMBEDDING_MODELS.index(last_emb)
            except ValueError: emb_idx = 0
                
            st.selectbox(MSG_EMBEDDING_SELECTOR_LABEL, AVAILABLE_EMBEDDING_MODELS, index=emb_idx, key="embedding_model_selector", on_change=embedding_selector_callback, disabled=is_generating)
        
        # --- 3. 시스템 상태 카드 ---
        with st.expander("📊 " + MSG_SYSTEM_STATUS_TITLE, expanded=True):
            status_placeholder = st.empty()
            SessionManager.set("status_placeholder", status_placeholder)
            _render_status_box(status_placeholder)

        return st.container()


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.fragment
def _pdf_viewer_fragment():
    st.subheader(MSG_PDF_VIEWER_TITLE)
    pdf_path = SessionManager.get("pdf_file_path")
    if not pdf_path:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return
    if not os.path.exists(pdf_path):
        st.error("⚠️ 파일을 찾을 수 없습니다.")
        return
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            if "current_page" not in st.session_state: st.session_state.current_page = 1
            is_generating = SessionManager.get("is_generating_answer")
            pdf_viewer(input=pdf_path, height=UI_CONTAINER_HEIGHT, pages_to_render=[st.session_state.current_page])
            def go_prev():
                if st.session_state.current_page > 1: st.session_state.current_page -= 1
            def go_next():
                if st.session_state.current_page < total_pages: st.session_state.current_page += 1
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: st.button(MSG_PDF_VIEWER_PREV_BUTTON, key="btn_pdf_prev", use_container_width=True, disabled=(st.session_state.current_page <= 1 or is_generating), on_click=go_prev)
            with c2:
                p1, p2 = st.columns([1, 1])
                with p1: st.number_input("페이지 이동", min_value=1, max_value=total_pages, value=st.session_state.current_page, label_visibility="collapsed", key="num_input_page", disabled=is_generating, on_change=lambda: setattr(st.session_state, 'current_page', int(st.session_state.num_input_page)))
                with p2: st.markdown(f"<div style='line-height: 2.3em; font-size: 1.0em;'>&nbsp;/ {total_pages} pages</div>", unsafe_allow_html=True)
            with c3: st.button(MSG_PDF_VIEWER_NEXT_BUTTON, key="btn_pdf_next", use_container_width=True, disabled=(st.session_state.current_page >= total_pages or is_generating), on_click=go_next)
    except Exception as e:
        st.error(f"PDF 오류: {e}")


def render_left_column():
    st.markdown("""
    <style>
    .tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #888; cursor: help; color: #0068c9; font-weight: bold; }
    .tooltip .tooltip-text { visibility: hidden; width: 350px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; font-size: 0.9em; font-weight: normal; line-height: 1.5; position: absolute; z-index: 1000; bottom: 125%; left: 50%; margin-left: -175px; opacity: 0; transition: opacity 0.3s; max-height: 200px; overflow-y: auto; box-shadow: 0px 4px 8px rgba(0,0,0,0.3); }
    .tooltip .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent; }
    .tooltip:hover .tooltip-text { visibility: visible; opacity: 1; }
    @media (prefers-color-scheme: dark) { .tooltip { color: #4fa8ff; } }
    </style>
    """, unsafe_allow_html=True)
    _chat_fragment()


def render_message(role: str, content: str):
    avatar_icon = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(content, unsafe_allow_html=True)


@st.fragment
def _chat_fragment():
    st.subheader(MSG_CHAT_TITLE)
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)
    messages = SessionManager.get_messages()
    
    # 1. 채팅 이력 렌더링
    for msg in messages:
        with chat_container: 
            render_message(msg["role"], msg["content"])
            
    if not messages:
        with chat_container: 
            st.info(MSG_CHAT_WELCOME)
            
    # 2. 사용자 입력 처리
    is_generating = SessionManager.get("is_generating_answer")
    if user_query := st.chat_input(MSG_CHAT_INPUT_PLACEHOLDER, disabled=is_generating, key="chat_input_clean"):
        SessionManager.add_message("user", user_query)
        SessionManager.add_status_log("질문 분석 중")
        
        # UI 즉시 업데이트
        status_placeholder = SessionManager.get("status_placeholder")
        _render_status_box(status_placeholder)
        with chat_container: 
            render_message("user", user_query)
            
        # RAG 엔진 호출
        rag_engine = SessionManager.get("rag_engine")
        if rag_engine:
            from common.utils import sync_run
            final_answer = sync_run(_stream_chat_response(rag_engine, user_query, chat_container))
            
            if final_answer and not final_answer.startswith("❌"):
                SessionManager.add_message("assistant", final_answer)
                SessionManager.replace_last_status_log("답변 작성 완료")
                SessionManager.add_status_log("질문 가능")
                st.rerun()
        else: 
            st.toast(MSG_CHAT_NO_QA_SYSTEM, icon="⚠️")
