"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import aclosing

import streamlit as st

from api.streaming_handler import get_adaptive_controller, get_streaming_handler
from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_PDF_VIEWER_NO_FILE,
    MSG_PREPARING_ANSWER,
    MSG_SYSTEM_STATUS_TITLE,
    UI_CONTAINER_HEIGHT,
)
from common.utils import apply_tooltips_to_response
from core.session import SessionManager
from ui.components.status_box import render_status_box as _render_status_box

logger = logging.getLogger(__name__)


async def _stream_chat_response(rag_engine, user_query: str, chat_container) -> str:
    """
    적응형 스트리밍 핸들러를 사용하여 사고 과정과 답변을 실시간으로 렌더링합니다.
    """

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

    # 핸들러 및 제어기 획득
    handler = get_streaming_handler()
    controller = get_adaptive_controller()

    # UI 디바운싱 설정 (루프 외부에서 정의하여 UnboundLocalError 방지)
    last_render_time = 0.0
    render_interval = 0.05  # 약 20fps로 UI 업데이트 제한

    try:
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                thought_container = st.empty()
                thought_display = None
                answer_display = st.empty()
                answer_display.markdown(f"⌛ {MSG_PREPARING_ANSWER}")

                # 적응형 스트리밍 적용 및 리소스 안전 관리
                event_generator = rag_engine.astream_events(
                    {"input": user_query}, config=run_config, version="v2"
                )

                async with aclosing(
                    handler.stream_graph_events(
                        event_generator, adaptive_controller=controller
                    )
                ) as stream:
                    async for chunk in stream:
                        # 상태 박스 동기화 (주기적: 오버헤드 감소를 위해 빈도 낮춤)
                        if chunk.chunk_index % 20 == 0:
                            _render_status_box(status_placeholder)

                        # 1. 메타데이터(문서) 처리
                        if chunk.metadata and "documents" in chunk.metadata:
                            state["retrieved_docs"] = chunk.metadata["documents"]

                        # 2. 사고 과정 처리
                        if chunk.thought:
                            if not state["full_thought"]:
                                state["thinking_start_time"] = time.time()
                                with thought_container:
                                    thought_expander = st.expander(
                                        "🧠 사고 과정 작성 중...", expanded=False
                                    )
                                    thought_display = thought_expander.empty()

                            state["full_thought"] += chunk.thought

                            # 사고 과정 디바운싱
                            current_time = time.time()
                            if current_time - last_render_time > render_interval:
                                if thought_display:
                                    thought_display.markdown(
                                        state["full_thought"] + "▌"
                                    )
                                last_render_time = current_time

                        # 3. 답변 본문 처리
                        if chunk.content:
                            if not state["full_response"]:
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

                            state["full_response"] += chunk.content

                            # UI 렌더링 시간 측정 및 디바운싱 적용
                            current_time = time.time()
                            if (
                                current_time - last_render_time > render_interval
                                or chunk.is_final
                            ):
                                render_start = current_time

                                # 성능 최적화: 스트리밍 중에는 무거운 수식 정규화를 건너뛰고 최종 결과에서만 수행
                                answer_display.markdown(
                                    state["full_response"] + "▌", unsafe_allow_html=True
                                )

                                # UI 렌더링 시간 기록 (적응형 제어용)
                                render_latency = (time.time() - render_start) * 1000
                                controller.record_latency(render_latency)
                                last_render_time = time.time()

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
    current_file_name: str | None = None,
    current_embedding_model: str | None = None,
    available_models: list[str] | None = None,
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

        # --- 3. 모델 설정 섹션 ---
        st.markdown("**⚙️ 모델 설정**")

        # 모델 목록 상태에 따른 선택창 렌더링 (사라짐 방지)
        from common.config import DEFAULT_OLLAMA_MODEL

        if available_models is None:
            # 로딩 중 상태 (고정된 위치)
            st.selectbox(
                "메인 LLM",
                ["모델 목록을 불러오는 중..."],
                index=0,
                disabled=True,
                key="model_selector_loading",
                label_visibility="collapsed",
            )
        else:
            # 로딩 완료 상태
            is_ollama_error = (
                available_models[0] == "Ollama 서버가 실행 중이지 않습니다."
                if available_models
                else False
            )
            actual_models = (
                []
                if is_ollama_error
                else [m for m in available_models if "---" not in m]
            )

            # 현재 선택된 모델 인덱스 계산
            last_model = SessionManager.get("last_selected_model")
            if not last_model or (actual_models and last_model not in actual_models):
                last_model = (
                    DEFAULT_OLLAMA_MODEL
                    if DEFAULT_OLLAMA_MODEL in actual_models
                    else (actual_models[0] if actual_models else available_models[0])
                )
                SessionManager.set("last_selected_model", last_model)

            try:
                model_idx = available_models.index(last_model)
            except ValueError:
                model_idx = 0

            st.selectbox(
                "메인 LLM",
                available_models,
                index=model_idx,
                key="model_selector",
                on_change=model_selector_callback,
                disabled=is_ollama_error or is_generating,
                label_visibility="collapsed",
            )

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
                disabled=is_generating or (available_models is None),
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

            # [추가] 딥 링크 요청 처리
            page_to_move = SessionManager.get("pdf_page_to_move")
            if page_to_move is not None:
                # 유효한 범위 내에서만 이동
                target = max(1, min(int(page_to_move), total_pages))
                st.session_state.current_page = target
                # 이동 후 요청 초기화
                SessionManager.set("pdf_page_to_move", None)

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
        cursor: pointer; 
        color: #0068c9; 
        font-weight: bold; 
        transition: all 0.2s;
        padding: 0 2px;
        border-radius: 4px;
    }
    .tooltip:hover {
        background-color: rgba(0, 104, 201, 0.1);
        color: #004a8b;
    }
    
    /* 인용 링크 기본 스타일 제거 */
    .citation-link {
        text-decoration: none !important;
        color: inherit !important;
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
            documents = [doc_pool[d_id] for d_id in doc_ids if d_id in doc_pool]

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
