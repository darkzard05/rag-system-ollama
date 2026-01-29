"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

from __future__ import annotations
import asyncio
import time
import logging
import os
import re
from contextlib import aclosing
from typing import Callable, Optional

import streamlit as st

from core.session import SessionManager
from common.utils import apply_tooltips_to_response
from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    UI_CONTAINER_HEIGHT,
    MSG_SYSTEM_STATUS_TITLE,
    MSG_PDF_VIEWER_NO_FILE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_PREPARING_ANSWER,
)

logger = logging.getLogger(__name__)


def _render_status_box(container):
    """시스템 상태 로그 박스를 최신순(역순)으로 렌더링합니다."""
    if container is None:
        return

    # [최적화] 세션이 없어도 에러 없이 빈 목록 반환
    try:
        status_logs = SessionManager.get("status_logs", [])
    except:
        status_logs = []

    if not status_logs:
        container.info("시스템 준비 중...")
        return

    # [스타일링: 최신순 출력 전용 테마]
    log_html = """
    <style>
    .status-outer-container {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 10px;
        background-color: rgba(128, 128, 128, 0.05);
        margin-bottom: 15px;
        width: 100%;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-container {
        font-family: 'Source Code Pro', 'Consolas', monospace;
        height: 140px;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .status-line {
        flex-shrink: 0;
        line-height: 1.5;
        margin: 0px !important;
        padding: 4px 8px !important;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #666;
        border-left: 2px solid transparent;
        transition: all 0.2s;
    }
    .status-newest { 
        color: #0068c9;
        font-weight: 600;
        background-color: rgba(0, 104, 201, 0.1);
        border-radius: 6px;
        border-left: 3px solid #0068c9;
    }
    
    @media (prefers-color-scheme: dark) {
        .status-outer-container { background-color: rgba(255, 255, 255, 0.05); }
        .status-line { color: #aaa; }
        .status-newest { color: #4fa8ff; background-color: rgba(79, 168, 255, 0.15); border-left-color: #4fa8ff; }
    }
    
    .status-container::-webkit-scrollbar { width: 4px; }
    .status-container::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.3); border-radius: 10px; }
    </style>
    """

    import html

    log_content = ""
    reversed_logs = status_logs[::-1]

    for i, log in enumerate(reversed_logs):
        safe_log = html.escape(log)
        clean_log = re.sub(
            r"[^\x00-\x7F가-힣\s\(\)\[\]\/\:\.\-\>]", "", safe_log
        ).strip()
        if not clean_log and safe_log:
            clean_log = safe_log.strip()

        is_newest = i == 0
        cls = "status-newest" if is_newest else ""
        icon = "●" if is_newest else "○"

        log_content += f"<div class='status-line {cls}' title='{clean_log}'>{icon} {clean_log}</div>"

    full_html = f"{log_html}<div class='status-outer-container'><div class='status-container'>{log_content}</div></div>"
    container.markdown(full_html, unsafe_allow_html=True)


async def _stream_chat_response(rag_engine, user_query: str, chat_container) -> str:
    """
    RAG 엔진의 이벤트를 수신하여 사고 과정과 답변을 실시간으로 렌더링합니다.
    """
    from common.utils import normalize_latex_delimiters  # 루프 밖으로 이동

    state = {
        "full_response": "",
        "full_thought": "",
        "retrieved_docs": [],
        "start_time": time.time(),
        "thinking_start_time": None,
        "thinking_end_time": None,
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
                # UI 컴포넌트 초기화: 공간만 확보하고 아무것도 표시하지 않음
                thought_container = st.empty()
                thought_display = None  # 사고 과정 텍스트를 실시간으로 쓸 공간

                answer_display = st.empty()
                answer_display.markdown(f"⌛ {MSG_PREPARING_ANSWER}")

                # 스트리밍 이벤트 수신
                async with aclosing(
                    rag_engine.astream_events(
                        {"input": user_query}, config=run_config, version="v2"
                    )
                ) as event_stream:
                    try:
                        async for event in event_stream:
                            kind, name, data = (
                                event["event"],
                                event.get("name", "Unknown"),
                                event.get("data", {}),
                            )

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
                                        # [개선] 실제 사고 토큰이 들어올 때만 익스팬더 생성
                                        with thought_container:
                                            thought_expander = st.expander(
                                                "🧠 사고 과정 작성 중...",
                                                expanded=False,
                                            )
                                            thought_display = thought_expander.empty()

                                    state["full_thought"] += thought
                                    if thought_display:
                                        thought_display.markdown(
                                            state["full_thought"] + "▌"
                                        )

                                # 2. 답변 본문 처리
                                if content:
                                    if not state["full_response"]:
                                        # 첫 답변 토큰이 들어오면 사고 과정 종료로 간주
                                        state["thinking_end_time"] = time.time()
                                        if state["full_thought"]:
                                            thinking_dur = (
                                                state["thinking_end_time"]
                                                - state["thinking_start_time"]
                                            )
                                            with thought_container:
                                                label = f"🧠 사고 완료 ({thinking_dur:.1f}초)"
                                                with st.expander(label, expanded=False):
                                                    st.markdown(state["full_thought"])

                                    state["full_response"] += content

                                    # [수정] 수식 구분자 실시간 정규화 적용
                                    display_text = normalize_latex_delimiters(
                                        state["full_response"]
                                    )
                                    answer_display.markdown(
                                        display_text + "▌", unsafe_allow_html=True
                                    )

                            # 엔진 내부 데이터 캡처
                            elif kind == "on_chain_end":
                                if name == "retrieve":
                                    output = data.get("output", {})
                                    if "documents" in output:
                                        state["retrieved_docs"] = output["documents"]

                                elif name == "generate_response":
                                    output = data.get("output", {})
                                    if isinstance(output, dict):
                                        if (
                                            "documents" in output
                                            and not state["retrieved_docs"]
                                        ):
                                            state["retrieved_docs"] = output[
                                                "documents"
                                            ]
                                        if "response" in output and len(
                                            output["response"]
                                        ) > len(state["full_response"]):
                                            state["full_response"] = output["response"]
                    except asyncio.CancelledError:
                        logger.info("[UI] 스트리밍이 사용자에 의해 중단되었습니다.")
                        raise

                # 최종 렌더링 및 정리
                _finalize_ui_rendering(thought_container, answer_display, state)

        return {
            "response": state["full_response"],
            "thought": state["full_thought"],
            "documents": state["retrieved_docs"],
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
            final_html = apply_tooltips_to_response(
                state["full_response"], state["retrieved_docs"]
            )
            answer_display.markdown(final_html, unsafe_allow_html=True)
        else:
            answer_display.markdown(state["full_response"], unsafe_allow_html=True)
    else:
        answer_display.error("⚠️ 답변이 생성되지 않았습니다.")


def render_sidebar(
    file_uploader_callback: Callable,
    model_selector_callback: Callable,
    embedding_selector_callback: Callable,
    is_generating: bool = False,
    current_file_name: Optional[str] = None,
    current_embedding_model: Optional[str] = None,
):
    # 커스텀 얇은 구분선 컴포넌트
    thin_divider = "<hr style='margin: 12px 0; border: none; border-top: 1px solid rgba(49, 51, 63, 0.1);'>"

    with st.sidebar:
        # --- 1. 브랜딩 섹션 (즉시 출력) ---
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                <span style='font-size: 2.2rem;'>🤖</span>
                <div>
                    <div style='font-size: 1.1rem; font-weight: bold; line-height: 1.2;'>GraphRAG-Ollama</div>
                    <div style='font-size: 0.75rem; color: #888;'>Local Intelligence RAG System</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(thin_divider, unsafe_allow_html=True)

        # --- 2. 문서 제어 섹션 ---
        st.markdown("**📄 문서 분석**")
        st.file_uploader(
            "PDF 파일 업로드",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            disabled=is_generating,
            label_visibility="collapsed",
        )

        if current_file_name:
            st.caption(f"현재: **{current_file_name}**")
        else:
            st.caption("분석할 PDF를 업로드하세요.")

        st.markdown(thin_divider, unsafe_allow_html=True)

        # --- 3. 모델 설정 섹션 (플레이스홀더) ---
        st.markdown("**⚙️ 모델 설정**")
        model_selector_placeholder = st.empty()

        with st.popover("🔧 고급 설정", use_container_width=True):
            st.markdown("#### 임베딩 설정")
            last_emb = current_embedding_model or AVAILABLE_EMBEDDING_MODELS[0]
            try:
                emb_idx = AVAILABLE_EMBEDDING_MODELS.index(last_emb)
            except ValueError:
                emb_idx = 0

            st.selectbox(
                "임베딩 모델",
                AVAILABLE_EMBEDDING_MODELS,
                index=emb_idx,
                key="embedding_model_selector",
                on_change=embedding_selector_callback,
                disabled=is_generating,
            )
            st.info("💡 하이브리드 검색 활성 중")

        st.markdown(thin_divider, unsafe_allow_html=True)

        # --- 4. 시스템 상태 섹션 ---
        st.markdown(f"**📊 {MSG_SYSTEM_STATUS_TITLE}**")
        status_placeholder = st.empty()

        # 상태 정보가 있을 때만 렌더링 (초기 렌더링 시에는 건너뜀)
        if "_initialized" in st.session_state:
            _render_status_box(status_placeholder)

        return {
            "model_selector": model_selector_placeholder,
            "status_container": status_placeholder,
        }


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.fragment
def _pdf_viewer_fragment():
    import fitz  # PyMuPDF
    from streamlit_pdf_viewer import pdf_viewer

    # [UI 대칭성] 채팅창과 동일하게 테두리가 있는 컨테이너 생성
    viewer_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    # [수정] 세션 초기화 전에도 안전하도록 기본값 None 제공 및 명시적 체크
    pdf_path_raw = SessionManager.get("pdf_file_path", None)

    if not pdf_path_raw:
        with viewer_container:
            st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    # [수정] 절대 경로로 변환하여 정확한 파일 참조 보장
    pdf_path = os.path.abspath(pdf_path_raw)

    if not os.path.exists(pdf_path):
        with viewer_container:
            st.error(f"⚠️ 파일을 찾을 수 없습니다: {pdf_path}")
        return

    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1

            # [수정] 세션 초기화 전에도 안전하도록 기본값 False 제공
            is_generating = SessionManager.get("is_generating_answer", False) or False

            # 1. PDF 뷰어 메인 영역
            with viewer_container:
                pdf_viewer(
                    input=pdf_path,
                    height=UI_CONTAINER_HEIGHT,
                    pages_to_render=[st.session_state.current_page],
                )

            # 2. 세련된 버튼 그룹형 탐색 툴바
            # 비율 조정: [이전|다음 | 페이지정보 | 슬라이더]
            c1, c2, c3, c4 = st.columns([4.0, 1.2, 0.4, 0.4])

            with c1:
                # 우측의 넓은 공간을 차지하는 슬라이더
                new_page = st.slider(
                    "page_nav_wide",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.current_page,
                    key="pdf_nav_slider_wide",
                    disabled=is_generating,
                    label_visibility="collapsed",
                )
                if new_page != st.session_state.current_page:
                    st.session_state.current_page = new_page
                    st.rerun()

            with c2:
                # 페이지 정보를 버튼 바로 옆에 배치
                st.markdown(
                    f"<div style='text-align: center; line-height: 2.3rem; font-family: monospace; font-size: 0.95rem; color: #888;'>"
                    f"<span style='color: #0068c9; font-weight: bold;'>{st.session_state.current_page}</span> / {total_pages}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c3:
                if st.button(
                    "‹",
                    use_container_width=True,
                    disabled=(st.session_state.current_page <= 1 or is_generating),
                    key="btn_pdf_prev_grp",
                    help="이전 페이지",
                ):
                    st.session_state.current_page -= 1
                    st.rerun()

            with c4:
                if st.button(
                    "›",
                    use_container_width=True,
                    disabled=(
                        st.session_state.current_page >= total_pages or is_generating
                    ),
                    key="btn_pdf_next_grp",
                    help="다음 페이지",
                ):
                    st.session_state.current_page += 1
                    st.rerun()

    except Exception as e:
        with viewer_container:
            st.error(f"PDF 오류: {e}")


def inject_custom_css():
    """앱 전반에 걸친 커스텀 CSS를 주입합니다."""
    # Streamlit 1.34+ 에서 지원하는 st.html 사용 (안전성 향상)
    st.html("""
    <style>
    /* Streamlit 기본 상태 표시기(Running...) 숨기기 */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
        display: none;
    }
    
    /* 툴팁 기본 스타일 */
    .tooltip { 
        position: relative; 
        display: inline-block; 
        border-bottom: 1px dotted #888; 
        cursor: help; 
        color: #0068c9; 
        font-weight: bold; 
    }
    .tooltip .tooltip-text { 
        visibility: hidden; 
        width: 350px; 
        background-color: #333; 
        color: #fff; 
        text-align: left; 
        border-radius: 8px; 
        padding: 12px; 
        font-size: 0.85rem; 
        font-weight: normal; 
        line-height: 1.5; 
        position: absolute; 
        z-index: 1000; 
        bottom: 125%; 
        left: 50%; 
        margin-left: -175px; 
        opacity: 0; 
        transition: opacity 0.3s, transform 0.3s; 
        transform: translateY(10px);
        max-height: 250px; 
        overflow-y: auto; 
        box-shadow: 0px 8px 16px rgba(0,0,0,0.4); 
        border: 1px solid #444;
    }
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
    .tooltip:hover .tooltip-text { 
        visibility: visible; 
        opacity: 1; 
        transform: translateY(0);
    }
    
    /* 다크 모드 대응 */
    @media (prefers-color-scheme: dark) { 
        .tooltip { color: #4fa8ff; } 
        .tooltip .tooltip-text { background-color: #262730; border-color: #444; }
        .tooltip .tooltip-text::after { border-color: #262730 transparent transparent transparent; }
    }
    
    /* 채팅 메시지 내 코드 블록 스타일 개선 */
    code {
        background-color: rgba(128, 128, 128, 0.15);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Source Code Pro', monospace;
    }
    
    /* PDF 컨트롤러 툴바 스타일 (더 세련된 버전) */
    .pdf-nav-container {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 4px 12px;
        margin-top: -8px;
        border: 1px solid rgba(49, 51, 63, 0.1);
        display: flex;
        align-items: center;
    }
    /* 슬라이더 높이 및 여백 조정 */
    div[data-testid="stSlider"] {
        padding-top: 10px;
        padding-bottom: 0px;
    }
    /* 버튼 스타일 미세 조정 */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(49, 51, 63, 0.1) !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: rgba(0, 104, 201, 0.1) !important;
        border-color: #0068c9 !important;
    }
    </style>
    """)


def render_left_column():
    _chat_fragment()


def render_message(role: str, content: str, thought: str = None, doc_ids: list = None):
    avatar_icon = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar_icon):
        if thought and thought.strip():
            with st.expander("🧠 사고 완료", expanded=False):
                st.markdown(thought)

        # [최적화] ID 리스트로부터 문서 풀에서 원본 문서 복원
        documents = []
        if role == "assistant" and doc_ids:
            # [수정] 세션 초기화 전에도 안전하도록 기본값 {} 제공
            doc_pool = SessionManager.get("doc_pool", {}) or {}
            documents = [doc_pool[d_id] for d in doc_ids if (d_id := d) in doc_pool]

        # Assistant 메시지이면서 참고 문서가 있다면 툴팁 적용
        if role == "assistant":
            from common.utils import (
                apply_tooltips_to_response,
                normalize_latex_delimiters,
            )

            # 1. 수식 정규화
            content = normalize_latex_delimiters(content)

            # 2. 툴팁 적용
            if documents:
                content = apply_tooltips_to_response(content, documents)

        st.markdown(content, unsafe_allow_html=True)


def _chat_fragment():
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)
    # [수정] 세션 초기화 전에도 안전하도록 기본값 [] 제공
    messages = SessionManager.get_messages() or []
    pdf_path = SessionManager.get("pdf_file_path")
    pdf_processed = SessionManager.get("pdf_processed", False)
    is_generating = bool(st.session_state.get("is_generating_answer", False))

    # 문서 분석 중인지 판별 (파일은 업로드됐는데 아직 처리가 안 된 상태)
    is_processing_pdf = bool(pdf_path and not pdf_processed)

    # 1. 채팅 이력 렌더링
    with chat_container:
        for msg in messages:
            render_message(
                msg["role"],
                msg["content"],
                thought=msg.get("thought"),
                doc_ids=msg.get("doc_ids"),
            )

        if not messages:
            if is_processing_pdf:
                st.info(
                    "📄 **문서를 분석하고 있습니다.**\n\n내용이 많을 경우 시간이 다소 소요될 수 있습니다. 완료 후 자동으로 채팅이 활성화됩니다."
                )
            else:
                st.info(MSG_CHAT_WELCOME)

    # 2. 사용자 입력 처리
    # 입력창 상태 결정
    input_disabled = is_generating or is_processing_pdf
    input_placeholder = (
        "문서 분석 중에는 질문할 수 없습니다..."
        if is_processing_pdf
        else (
            "답변을 생성하는 중입니다..."
            if is_generating
            else MSG_CHAT_INPUT_PLACEHOLDER
        )
    )

    if user_query := st.chat_input(
        input_placeholder, disabled=input_disabled, key="chat_input_clean"
    ):
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

            result = sync_run(
                _stream_chat_response(rag_engine, user_query, chat_container)
            )

            final_answer = result.get("response", "")
            final_thought = result.get("thought", "")
            final_docs = result.get("documents", [])

            if final_answer and not final_answer.startswith("❌"):
                SessionManager.add_message(
                    "assistant",
                    final_answer,
                    thought=final_thought,
                    documents=final_docs,  # SessionManager.add_message 내부에서 doc_ids로 변환됨
                )
                SessionManager.replace_last_status_log("답변 작성 완료")
                SessionManager.add_status_log("질문 가능")
                st.rerun()
        else:
            st.toast(MSG_CHAT_NO_QA_SYSTEM, icon="⚠️")
