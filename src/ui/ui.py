"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

import asyncio
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
    DEFAULT_OLLAMA_MODEL,
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
    """시스템 상태 로그 박스를 최신순(역순)으로 렌더링합니다."""
    if container is None:
        return
        
    status_logs = SessionManager.get("status_logs", [])
    if not status_logs:
        return

    # [스타일링: 최신순 출력 전용 테마]
    log_html = """
    <style>
    .status-outer-container {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 8px;
        padding: 8px;
        background-color: #1e1e1e;
        margin-bottom: 10px;
        width: 100%;
    }
    .status-container {
        font-family: 'Consolas', 'Monaco', 'Source Code Pro', monospace;
        height: 130px;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column; /* 역순 데이터이므로 위에서부터 순차 출력 */
        gap: 2px;
    }
    .status-line {
        flex-shrink: 0;
        line-height: 1.4;
        margin: 0px !important;
        padding: 2px 6px !important;
        font-size: 0.82rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #888; /* 기본은 흐리게 */
    }
    .status-newest { 
        color: #4fc3f7; /* 최신 로그만 밝은 색 */
        font-weight: bold;
        background-color: rgba(79, 195, 247, 0.08);
        border-radius: 4px;
    }
    
    .status-container::-webkit-scrollbar { width: 3px; }
    .status-container::-webkit-scrollbar-thumb { background: #333; }
    </style>
    """
    
    import re
    import html
    log_content = ""
    # [핵심] 로그를 역순으로 뒤집어 최신 내용이 0번 인덱스에 오게 함
    reversed_logs = status_logs[::-1]
    
    for i, log in enumerate(reversed_logs):
        # HTML 이스케이프 처리로 안전성 확보
        safe_log = html.escape(log)
        clean_log = re.sub(r'[^\x00-\x7F가-힣\s\(\)\[\]\/\:\.\-\>]', '', safe_log).strip()
        if not clean_log and safe_log: clean_log = safe_log.strip()
        
        # 첫 번째(i=0)가 가장 최신 로그
        is_newest = (i == 0)
        cls = "status-newest" if is_newest else ""
        prefix = "⚡" if is_newest else " "
        
        log_content += f"<div class='status-line {cls}' title='{clean_log}'>{prefix} {clean_log}</div>"
    
    full_html = f"{log_html}<div class='status-outer-container'><div class='status-container'>{log_content}</div></div>"
    container.markdown(full_html, unsafe_allow_html=True)



async def _stream_chat_response(rag_engine, user_query: str, chat_container) -> str:
    """
    RAG 엔진의 이벤트를 수신하여 사고 과정과 답변을 실시간으로 렌더링합니다.
    """
    state = {
        "full_response": "",
        "full_thought": "",
        "retrieved_docs": [],
        "start_time": time.time(),
        "thinking_start_time": None,
        "thinking_end_time": None
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
                # UI 컴포넌트 초기화: 처음에는 접힌 상태로 대기
                thought_container = st.empty()
                with thought_container:
                    st.expander("🧠 사고 준비 중...", expanded=False)
                
                thought_display = None # 사고 과정 텍스트를 실시간으로 쓸 공간
                
                answer_display = st.empty()
                answer_display.markdown(f"⌛ {MSG_PREPARING_ANSWER}")
                
                # 스트리밍 이벤트 수신
                async with aclosing(rag_engine.astream_events(
                    {"input": user_query}, config=run_config, version="v2"
                )) as event_stream:
                    try:
                        async for event in event_stream:
                            kind, name, data = event["event"], event.get("name", "Unknown"), event.get("data", {})
                            
                            # 상태 박스 동기화
                            if kind in ["on_chain_start", "on_chain_end"]:
                                _render_status_box(status_placeholder)
                            
                            # 커스텀 응답 이벤트 처리 (Integrity Protocol)
                            if kind == "on_custom_event" and name == "response_chunk":
                                content = data.get("chunk")
                                thought = data.get("thought")

                                # 1. 사고 과정 처리
                                if thought:
                                    if not state["full_thought"]:
                                        state["thinking_start_time"] = time.time()
                                        # 사고 시작 시 타이틀 업데이트 (여전히 접힌 상태 유지)
                                        with thought_container:
                                            thought_expander = st.expander("🧠 사고 과정 작성 중...", expanded=False)
                                            thought_display = thought_expander.empty()
                                    
                                    state["full_thought"] += thought
                                    if thought_display:
                                        thought_display.markdown(state["full_thought"] + "▌")
                                
                                # 2. 답변 본문 처리
                                if content:
                                    if not state["full_response"]:
                                        # 첫 답변 토큰이 들어오면 사고 과정 종료로 간주
                                        state["thinking_end_time"] = time.time()
                                        if state["full_thought"]:
                                            thinking_dur = state["thinking_end_time"] - state["thinking_start_time"]
                                            with thought_container:
                                                label = f"🧠 사고 완료 ({thinking_dur:.1f}초)"
                                                with st.expander(label, expanded=False):
                                                    st.markdown(state["full_thought"])
                                    
                                    state["full_response"] += content
                                    answer_display.markdown(state["full_response"] + "▌", unsafe_allow_html=True)
                                
                            # 엔진 내부 데이터 캡처
                            elif kind == "on_chain_end":
                                if name == "retrieve":
                                    output = data.get("output", {})
                                    if "documents" in output: state["retrieved_docs"] = output["documents"]
                                
                                elif name == "generate_response":
                                    output = data.get("output", {})
                                    if isinstance(output, dict):
                                        if "documents" in output and not state["retrieved_docs"]:
                                            state["retrieved_docs"] = output["documents"]
                                        if "response" in output and len(output["response"]) > len(state["full_response"]):
                                            state["full_response"] = output["response"]
                    except asyncio.CancelledError:
                        logger.info("[UI] 스트리밍이 사용자에 의해 중단되었습니다.")
                        raise
                
                # 최종 렌더링 및 정리
                _finalize_ui_rendering(thought_container, answer_display, state)
                
        return {
            "response": state["full_response"], 
            "thought": state["full_thought"],
            "documents": state["retrieved_docs"]
        }

    except Exception as e:
        logger.error(f"UI 스트리밍 오류: {e}", exc_info=True)
        from common.utils import format_error_message
        friendly_msg = format_error_message(e)
        
        # [최적화] 상태창에 에러 메시지 즉시 반영
        SessionManager.add_status_log(friendly_msg)
        return {"response": friendly_msg, "thought": "", "documents": []}
    finally:
        SessionManager.set("is_generating_answer", False)
        _render_status_box(status_placeholder)

def _finalize_ui_rendering(thought_container, answer_display, state):
    """답변 생성이 끝난 후 UI를 최종 상태로 정리합니다."""
    # 1. 사고 과정 정리
    if state["full_thought"]:
        with thought_container:
            # 타이밍 정보가 있으면 사용, 없으면 토큰 수 사용
            if state.get("thinking_start_time") and state.get("thinking_end_time"):
                dur = state["thinking_end_time"] - state["thinking_start_time"]
                label = f"🧠 사고 완료 ({dur:.1f}초)"
            else:
                label = f"🧠 사고 완료 ({len(state['full_thought'].split())} tokens)"
            
            with st.expander(label, expanded=False):
                st.markdown(state["full_thought"])
    else:
        thought_container.empty()

    # 2. 답변 본문 최종 렌더링 (툴팁 및 하이라이트 적용)
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
            
            # [수정] 저장된 세션 모델이 없거나 유효하지 않은 경우의 초기값 결정 로직
            if not last_model or (actual_models and last_model not in actual_models):
                # 1. 설정파일의 기본 모델(DEFAULT_OLLAMA_MODEL)이 목록에 있는지 확인
                if DEFAULT_OLLAMA_MODEL in actual_models:
                    last_model = DEFAULT_OLLAMA_MODEL
                # 2. 없다면 목록의 첫 번째 모델 선택
                elif actual_models:
                    last_model = actual_models[0]
                # 3. 목록도 없다면 상수의 기본값 사용
                else:
                    last_model = DEFAULT_OLLAMA_MODEL
                
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


def render_message(role: str, content: str, thought: str = None, doc_ids: list = None):
    avatar_icon = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar_icon):
        if thought:
            with st.expander("🧠 사고 완료", expanded=False):
                st.markdown(thought)
        
        # [최적화] ID 리스트로부터 문서 풀에서 원본 문서 복원
        documents = []
        if role == "assistant" and doc_ids:
            doc_pool = SessionManager.get("doc_pool", {})
            documents = [doc_pool[d_id] for d in doc_ids if (d_id := d) in doc_pool]
        
        # Assistant 메시지이면서 참고 문서가 있다면 툴팁 적용
        if role == "assistant" and documents:
            from common.utils import apply_tooltips_to_response
            content = apply_tooltips_to_response(content, documents)
            
        st.markdown(content, unsafe_allow_html=True)


@st.fragment
def _chat_fragment():
    st.subheader(MSG_CHAT_TITLE)
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)
    messages = SessionManager.get_messages()
    
    # 1. 채팅 이력 렌더링
    for msg in messages:
        with chat_container: 
            render_message(
                msg["role"], 
                msg["content"], 
                thought=msg.get("thought"),
                doc_ids=msg.get("doc_ids") # documents 대신 doc_ids 전달
            )
            
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
            result = sync_run(_stream_chat_response(rag_engine, user_query, chat_container))
            
            final_answer = result.get("response", "")
            final_thought = result.get("thought", "")
            final_docs = result.get("documents", [])
            
            if final_answer and not final_answer.startswith("❌"):
                SessionManager.add_message(
                    "assistant", 
                    final_answer, 
                    thought=final_thought,
                    documents=final_docs # SessionManager.add_message 내부에서 doc_ids로 변환됨
                )
                SessionManager.replace_last_status_log("답변 작성 완료")
                SessionManager.add_status_log("질문 가능")
                st.rerun()
        else: 
            st.toast(MSG_CHAT_NO_QA_SYSTEM, icon="⚠️")
