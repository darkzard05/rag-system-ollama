"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

import time
import logging
from typing import Callable, Optional

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF

from session import SessionManager
from model_loader import get_available_models
from utils import sync_run, apply_tooltips_to_response
from config import (
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


async def _stream_chat_response(qa_chain, user_input: str, chat_container) -> str:
    """
    최적화된 답변 생성 및 스트리밍 함수.
    [최적화] 인위적인 딜레이(update_interval)를 제거하여 반응 속도 극대화.
    [개선] 문서 출처를 RAG 결과에서 직접 가져와 구조적으로 렌더링.
    """
    full_response = ""
    retrieved_documents = [] # 검색된 문서 저장용
    start_time = time.time()
    
    current_llm = SessionManager.get("llm")
    if not current_llm:
        return "❌ 오류: LLM 모델이 로드되지 않았습니다."

    run_config = {"configurable": {"llm": current_llm}}
    SessionManager.set("is_generating_answer", True)

    try:
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                answer_container = st.empty()
                # 로딩 메시지 복구
                answer_container.markdown(f"⌛ {MSG_PREPARING_ANSWER}")

                # 스트리밍 루프
                async for event in qa_chain.astream_events(
                    {"input": user_input},
                    config=run_config,
                    version="v1"
                ):
                    kind = event["event"]
                    name = event.get("name", "Unknown")
                    data = event.get("data", {})
                    
                    # [Debug]
                    # logger.debug(f"[Stream Event] Kind: {kind} | Name: {name}")

                    chunk_text = None

                    # 1. 텍스트 스트리밍
                    if kind == "on_parser_stream":
                        chunk = data.get("chunk")
                        if isinstance(chunk, str):
                            chunk_text = chunk
                    
                    elif kind == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if hasattr(chunk, "content"):
                            chunk_text = chunk.content
                        elif isinstance(chunk, dict) and "content" in chunk:
                            chunk_text = chunk["content"]
                        elif isinstance(chunk, str):
                            chunk_text = chunk
                    
                    elif kind == "on_custom_event" and name == "response_chunk":
                        chunk = data.get("chunk")
                        if isinstance(chunk, str):
                            chunk_text = chunk
                    
                    if chunk_text:
                        full_response += chunk_text
                        answer_container.markdown(full_response + "▌", unsafe_allow_html=True)

                    # 2. 문서 캡처 (retrieve 노드 완료 시점)
                    if kind == "on_chain_end" and name == "retrieve":
                        output = data.get("output")
                        if output and isinstance(output, dict) and "documents" in output:
                            retrieved_documents = output["documents"]
                            logger.info(f"[UI] 검색된 문서 캡처: {len(retrieved_documents)}개")

                    # 3. 최종 결과 보정 (generate_response 노드 완료 시점)
                    if kind == "on_chain_end" and name == "generate_response":
                        output = data.get("output")
                        if isinstance(output, dict):
                            # 만약 노드가 documents를 반환한다면 여기서도 캡처 시도 (안전망)
                            if "documents" in output and not retrieved_documents:
                                retrieved_documents = output["documents"]
                            
                            if "response" in output:
                                final_node_res = output["response"]
                                if len(final_node_res) > len(full_response):
                                    full_response = final_node_res

                # 스트리밍 종료 후 처리
                elapsed_time = time.time() - start_time
                
                if full_response:
                    # [변경] 하단 목록 추가 대신 본문에 툴팁 적용
                    if retrieved_documents:
                        # 툴팁이 적용된 HTML로 변환
                        final_html = apply_tooltips_to_response(full_response, retrieved_documents)
                        answer_container.markdown(final_html, unsafe_allow_html=True)
                        
                        # 채팅 히스토리 저장을 위해 원본 텍스트(full_response)가 아닌 
                        # HTML 버전(final_html)을 반환해야 나중에도 툴팁이 보임.
                        # 단, SessionManager에는 구조적 데이터가 없으므로 HTML을 저장해야 함.
                        full_response = final_html 
                    else:
                        # 문서가 없으면 그냥 텍스트 렌더링
                        answer_container.markdown(full_response, unsafe_allow_html=True)

                    logger.info(f"[UI] 답변 생성 완료: {elapsed_time:.2f}초")
                else:
                    logger.warning(f"[UI] 답변 생성 실패. 소요시간: {elapsed_time:.2f}초")
                    answer_container.error("⚠️ 답변이 생성되지 않았습니다.")

        return full_response

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        return f"❌ 오류 발생: {str(e)}"
    finally:
        SessionManager.set("is_generating_answer", False)


def render_sidebar(
    file_uploader_callback: Callable,
    model_selector_callback: Callable,
    embedding_selector_callback: Callable
):
    """
    최소한의 설정만 남긴 사이드바를 렌더링합니다.
    공간 효율을 위해 구분선을 제거하고 Expander를 활용합니다.

    Args:
        file_uploader_callback: 파일 업로드 시 실행될 콜백 함수
        model_selector_callback: LLM 모델 변경 시 실행될 콜백 함수
        embedding_selector_callback: 임베딩 모델 변경 시 실행될 콜백 함수
    """
    with st.sidebar:
        st.header(MSG_SIDEBAR_TITLE)

        # 답변 생성 중인지 확인 (사이드바 전체 잠금용)
        is_generating = SessionManager.get("is_generating_answer")
        
        # 1. 파일 업로드 섹션 (가장 중요하므로 상시 노출)
        st.file_uploader(
            MSG_PDF_UPLOADER_LABEL, 
            type="pdf", 
            key="pdf_uploader", 
            on_change=file_uploader_callback,
            disabled=is_generating  # 생성 중 업로드 방지
        )

        # 2. 모델 설정 섹션 (접이식으로 공간 절약)
        # expanded=False로 설정하여 기본적으로는 숨김 처리 (사용자 경험에 따라 변경 가능)
        with st.expander("⚙️ 모델 설정", expanded=False):
            # 모델 목록 가져오기
            available_models = get_available_models()
            is_ollama_error = (
                bool(available_models) and 
                available_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING
            )
            actual_models = [] if is_ollama_error else [m for m in available_models if "---" not in m]
            
            last_model = SessionManager.get("last_selected_model")
            if not last_model or (actual_models and last_model not in actual_models):
                last_model = actual_models[0] if actual_models else OLLAMA_MODEL_NAME
                SessionManager.set("last_selected_model", last_model)

            try:
                idx = available_models.index(last_model)
            except ValueError:
                idx = 0

            # 긴 모델 이름이 잘리는 문제를 위해 help 툴팁 추가
            st.selectbox(
                MSG_MODEL_SELECTOR_LABEL, 
                available_models, 
                index=idx, 
                key="model_selector", 
                on_change=model_selector_callback, 
                disabled=(is_ollama_error or is_generating), # 에러거나 생성 중이면 비활성
                help="사용할 LLM 모델을 선택하세요."
            )

            # 임베딩 모델 선택
            last_emb = SessionManager.get("last_selected_embedding_model") or AVAILABLE_EMBEDDING_MODELS[0]
            try:
                emb_idx = AVAILABLE_EMBEDDING_MODELS.index(last_emb)
            except ValueError:
                emb_idx = 0
                
            st.selectbox(
                MSG_EMBEDDING_SELECTOR_LABEL, 
                AVAILABLE_EMBEDDING_MODELS, 
                index=emb_idx, 
                key="embedding_model_selector", 
                on_change=embedding_selector_callback,
                disabled=is_generating, # 생성 중 변경 방지
                help="문서 검색에 사용할 임베딩 모델입니다."
            )
        
        # 3. 시스템 상태 섹션 (구분선 없이 여백으로 분리)
        st.markdown("#### " + MSG_SYSTEM_STATUS_TITLE)
        
        # 상태 메시지를 표시할 빈 컨테이너 반환
        return st.container()


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.fragment
def _pdf_viewer_fragment():
    """PDF 뷰어 (Fragment) - 개선된 네비게이션"""
    st.subheader(MSG_PDF_VIEWER_TITLE)
    
    pdf_bytes = SessionManager.get("pdf_file_bytes")
    if not pdf_bytes:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return
    
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            
            # 세션 상태 초기화
            if "current_page" not in st.session_state: 
                st.session_state.current_page = 1
            
            # 페이지 범위 보정
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1
            if st.session_state.current_page < 1:
                st.session_state.current_page = 1

            # 답변 생성 중인지 확인
            is_generating = SessionManager.get("is_generating_answer")

            # --- PDF 뷰어 렌더링 ---
            pdf_viewer(
                input=pdf_bytes, 
                height=UI_CONTAINER_HEIGHT, 
                pages_to_render=[st.session_state.current_page]
            )

            # --- 하단 네비게이션 ---
            # 헬퍼 함수: 페이지 변경 콜백
            def go_prev():
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
            
            def go_next():
                if st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1

            # 레이아웃: [이전] [페이지 입력 / 총페이지] [다음]
            c1, c2, c3 = st.columns([1, 1, 1])
            
            with c1:
                # 이전 페이지 버튼 (on_click 사용)
                st.button(
                    MSG_PDF_VIEWER_PREV_BUTTON, 
                    key="btn_pdf_prev", 
                    use_container_width=True,
                    disabled=(st.session_state.current_page <= 1 or is_generating),
                    on_click=go_prev
                )

            with c2:
                # 가운데 병합된 '현재 / 총 page' 레이아웃
                p1, p2 = st.columns([1, 1])
                with p1:
                    def update_page_input():
                        # number_input의 값이 변경되면 session_state에 반영
                        # key가 'num_input_page'이므로 st.session_state.num_input_page에 값이 있음
                        st.session_state.current_page = int(st.session_state.num_input_page)

                    st.number_input(
                        "페이지 이동", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=st.session_state.current_page, 
                        label_visibility="collapsed",
                        key="num_input_page",
                        disabled=is_generating,
                        on_change=update_page_input
                    )
                with p2:
                    st.markdown(
                        f"<div style='line-height: 2.3em; font-size: 1.0em;'>"
                        f"&nbsp;/ {total_pages} pages</div>", 
                        unsafe_allow_html=True
                    )

            with c3:
                # 다음 페이지 버튼 (on_click 사용)
                st.button(
                    MSG_PDF_VIEWER_NEXT_BUTTON, 
                    key="btn_pdf_next", 
                    use_container_width=True,
                    disabled=(st.session_state.current_page >= total_pages or is_generating),
                    on_click=go_next
                )
            
    except Exception as e:
        logger.error(f"PDF 뷰어 오류: {e}", exc_info=True)
        st.error(f"PDF 오류: {e}")


def render_left_column():
    # [툴팁 CSS 주입]
    st.markdown("""
    <style>
    /* 툴팁 컨테이너 */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #888; /* 인용구임을 표시하는 점선 밑줄 */
        cursor: help;
        color: #0068c9; /* 링크 색상과 유사하게 */
        font-weight: bold;
    }

    /* 툴팁 텍스트 (숨김 상태) */
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 350px;
        background-color: #333; /* 다크 그레이 배경 */
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        font-size: 0.9em;
        font-weight: normal;
        line-height: 1.5;
        
        /* 위치 조정 */
        position: absolute;
        z-index: 1000;
        bottom: 125%; /* 텍스트 위쪽에 표시 */
        left: 50%;
        margin-left: -175px; /* 가운데 정렬 */
        
        /* 페이드인 효과 */
        opacity: 0;
        transition: opacity 0.3s;
        
        /* 스크롤 및 크기 제한 */
        max-height: 200px;
        overflow-y: auto;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    }

    /* 화살표 */
    .tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }

    /* 호버 시 표시 */
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* 다크모드 대응: 글자색을 명확히 */
    @media (prefers-color-scheme: dark) {
        .tooltip { color: #4fa8ff; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    _chat_fragment()


def render_message(role: str, content: str):
    """
    메시지를 역할에 따라 스타일링하여 렌더링합니다.
    Args:
        role: 메시지 작성자 역할 ('user' 또는 'assistant')
        content: 메시지 본문
    """
    # 아바타 설정: 사용자(👤), 어시스턴트(🤖)
    avatar_icon = "🤖" if role == "assistant" else "👤"
    
    with st.chat_message(role, avatar=avatar_icon):
        # [개선] 파싱 로직 제거 - content에 이미 포맷팅된 출처가 포함되어 있음
        # HTML 태그(접이식 출처) 렌더링을 위해 unsafe_allow_html=True 설정
        st.markdown(content, unsafe_allow_html=True)


@st.fragment
def _chat_fragment():
    """채팅 구역 (Fragment)"""
    st.subheader(MSG_CHAT_TITLE)
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    messages = SessionManager.get_messages()
    for msg in messages:
        with chat_container:
            render_message(msg["role"], msg["content"])

    if not messages:
        with chat_container: 
            st.info(MSG_CHAT_WELCOME)

    is_gen = SessionManager.get("is_generating_answer")
    if user_input := st.chat_input(MSG_CHAT_INPUT_PLACEHOLDER, disabled=is_gen, key="chat_input_clean"):
        SessionManager.add_message("user", user_input)
        
        # 즉시 사용자 메시지 렌더링
        with chat_container:
            render_message("user", user_input)

        qa_chain = SessionManager.get("qa_chain")
        if qa_chain:
            final_ans = sync_run(_stream_chat_response(qa_chain, user_input, chat_container))
            if final_ans and not final_ans.startswith("❌"):
                SessionManager.add_message("assistant", final_ans)
                st.rerun()
        else:
            st.toast(MSG_CHAT_NO_QA_SYSTEM, icon="⚠️")