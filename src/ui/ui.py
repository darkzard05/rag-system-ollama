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
from typing import Any

import streamlit as st

from api.streaming_handler import get_adaptive_controller, get_streaming_handler
from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_PDF_VIEWER_NO_FILE,
    UI_CONTAINER_HEIGHT,
)
from core.session import SessionManager

logger = logging.getLogger(__name__)


async def _stream_chat_response(
    rag_engine, user_query: str, chat_container
) -> dict[str, Any]:
    """
    적응형 스트리밍 핸들러를 사용하여 사고 과정과 답변을 실시간으로 렌더링합니다.
    """

    state: dict[str, Any] = {
        "full_response": "",
        "full_thought": "",
        "retrieved_docs": [],
        "start_time": time.time(),
        "thinking_start_time": None,
        "thinking_end_time": None,
    }

    current_llm = SessionManager.get("llm")
    if not current_llm:
        return {
            "response": "❌ 오류: 추론 모델이 로드되지 않았습니다.",
            "thought": "",
            "documents": [],
        }

    run_config = {"configurable": {"llm": current_llm}}
    SessionManager.set("is_generating_answer", True)

    # 핸들러 및 제어기 획득
    handler = get_streaming_handler()
    controller = get_adaptive_controller()

    # UI 디바운싱 설정
    last_render_time = 0.0
    render_interval = 0.05

    # 스트리밍 응답 렌더링
    try:
        with chat_container, st.chat_message("assistant", avatar="🤖"):
            # 파이프라인 상태 표시 (메시지 최상단에 텍스트 로그로 누적)
            status_container = st.empty()
            pipeline_logs = []

            def update_pipeline_display(new_log: str):
                pipeline_logs.append(f"└─ `PROCESS` {new_log}")
                status_container.markdown("  \n".join(pipeline_logs))

            # 사고 과정 및 답변 표시용 컨테이너
            thought_area = st.container()
            answer_area = st.empty()

            # 적응형 스트리밍 적용
            event_generator = rag_engine.astream_events(
                {"input": user_query}, config=run_config, version="v2"
            )

            async with aclosing(
                handler.stream_graph_events(
                    event_generator, adaptive_controller=controller
                )
            ) as stream:
                async for chunk in stream:
                    # A. 상태 업데이트 처리 (누적 로그 방식)
                    if chunk.status:
                        update_pipeline_display(chunk.status)
                        SessionManager.add_status_log(chunk.status)

                    # B. 메타데이터(문서) 처리
                    if chunk.metadata and "documents" in chunk.metadata:
                        state["retrieved_docs"] = chunk.metadata["documents"]
                        doc_count = len(state["retrieved_docs"])
                        update_pipeline_display(f"관련 지식 {doc_count}개 확보 완료")

                    # C. 사고 과정 처리
                    if chunk.thought:
                        # 사고 과정 시작 시 타이밍 기록
                        if not state["full_thought"]:
                            state["thinking_start_time"] = time.time()
                            with thought_area:
                                st.caption("AI의 사고 흐름:")
                                thought_display = st.empty()

                        state["full_thought"] += chunk.thought

                        current_time = time.time()
                        if current_time - last_render_time > render_interval:
                            thought_display.markdown(f"*{state['full_thought']}*")
                            last_render_time = current_time

                    # D. 답변 본문 처리
                    if chunk.content:
                        # 첫 토큰 수신 시 파이프라인 로그 정리 및 답변 시작
                        if not state["full_response"]:
                            status_container.empty()  # 진행 로그 제거 (답변 집중)
                            state["thinking_end_time"] = time.time()

                            # 사고 과정이 있었다면 예쁘게 마무리
                            if state["full_thought"]:
                                thinking_dur = (
                                    state["thinking_end_time"]
                                    - state["thinking_start_time"]
                                )
                                with thought_area:
                                    with st.expander(
                                        f"💭 사고 완료 ({thinking_dur:.1f}초)",
                                        expanded=False,
                                    ):
                                        st.markdown(state["full_thought"])
                                    if "thought_display" in locals():
                                        thought_display.empty()

                        state["full_response"] += chunk.content

                        current_time = time.time()
                        if (
                            current_time - last_render_time > render_interval
                            or chunk.is_final
                        ):
                            render_start = current_time
                            answer_area.markdown(state["full_response"] + "▌")

                            # 렌더링 성능 피드백
                            render_latency = (time.time() - render_start) * 1000
                            controller.record_latency(render_latency)
                            last_render_time = time.time()

            # 2. 최종 정돈 (인용구, 피드백 등)
            _finalize_ui_rendering(thought_area, answer_area, state)

            # 성능 메트릭 계산
            total_dur = time.time() - state["start_time"]
            token_count = len(state["full_response"].split())
            tps = token_count / total_dur if total_dur > 0 else 0

            metrics = {
                "duration": total_dur,
                "tps": tps,
                "doc_count": len(state["retrieved_docs"]),
                "model": SessionManager.get("last_selected_model", "Unknown"),
            }

        return {
            "response": state["full_response"],
            "thought": state["full_thought"],
            "documents": state["retrieved_docs"],
            "metrics": metrics if state["full_response"] else None,
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


def _finalize_ui_rendering(thought_container, answer_display, state):
    """답변 생성이 끝난 후 UI를 최종 상태로 정리합니다."""
    # 1. 사고 과정 정리
    if state["full_thought"]:
        with thought_container:
            if state.get("thinking_start_time") and state.get("thinking_end_time"):
                dur = state["thinking_end_time"] - state["thinking_start_time"]
                label = f"🧠 사고 완료 ({dur:.1f}초)"
            else:
                label = f"🧠 사고 완료 ({len(state['full_thought'].split())} tokens)"

            with st.expander(label, expanded=False):
                st.markdown(
                    f'<div class="thought-container">{state["full_thought"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        thought_container.empty()

    # 2. 답변 본문 최종 렌더링
    if state["full_response"]:
        from common.utils import apply_tooltips_to_response

        if state["retrieved_docs"]:
            final_html = apply_tooltips_to_response(
                state["full_response"], state["retrieved_docs"]
            )
            answer_display.markdown(final_html, unsafe_allow_html=True)
        else:
            answer_display.markdown(state["full_response"], unsafe_allow_html=True)

        # 2. 지능형 출처 표시 (st.pills & st.popover)
        if state["retrieved_docs"]:
            st.divider()
            c1, c2 = st.columns([0.85, 0.15])

            with c1:
                # 중복 제거 및 페이지 정렬 (이동을 위해 페이지 번호 저장)
                unique_sources = {}
                for doc in state["retrieved_docs"]:
                    src = doc.metadata.get("source", "Unknown")
                    pg = doc.metadata.get("page", "?")
                    key = f"📄 {src} (p.{pg})"
                    if key not in unique_sources:
                        unique_sources[key] = pg

            pill_key = f"pills_{int(state['start_time'])}"
            
            # [수정] 위젯 생성 전 상태를 먼저 체크하고 처리하여 StreamlitAPIException 방지
            if pill_key in st.session_state and st.session_state[pill_key]:
                selection = st.session_state[pill_key]
                target_pg = unique_sources.get(selection)
                if target_pg and str(target_pg).isdigit():
                    new_pg = int(target_pg)
                    st.session_state.current_page = new_pg
                    # 슬라이더 위젯 상태도 강제 업데이트하여 동기화
                    st.session_state.pdf_nav_slider_wide = new_pg
                    # 상태 초기화 (위젯 생성 전이므로 안전)
                    st.session_state[pill_key] = None
                    st.rerun()

            # st.pills를 활용한 칩 기반 인터페이스
            selected_pill = st.pills(
                "📍 참고 지식 (클릭 시 이동):",
                options=list(unique_sources.keys()),
                selection_mode="single",
                key=pill_key,
            )

            with c2:
                # 피드백 위젯
                st.feedback("thumbs", key=f"fb_{int(state['start_time'])}")

        # 3. 하단 메트릭 캡션
        total_dur = time.time() - state["start_time"]
        token_count = len(state["full_response"].split())
        tps = token_count / total_dur if total_dur > 0 else 0
        doc_count = len(state["retrieved_docs"])
        current_model = SessionManager.get("last_selected_model", "Unknown")

        # 표준 캡션 사용
        st.caption(
            f"⏱️ {total_dur:.1f}s | 🚀 {tps:.1f} t/s | 📄 {doc_count} refs | 🤖 {current_model}"
        )
    else:
        answer_display.error("⚠️ 답변이 생성되지 않았습니다.")


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.fragment
def _pdf_viewer_fragment():
    import fitz  # PyMuPDF
    from streamlit_pdf_viewer import pdf_viewer

    # 1. 이미 계산된 높이 가져오기 (폴백 800)
    win_h = st.session_state.get("last_valid_height", 800)
    container_h = max(400, win_h - 250)

    # [UI 대칭성] 채팅창과 동일하게 테두리가 있는 컨테이너 생성
    viewer_container = st.container(height=container_h, border=True)

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

            # [수정] 세션 초기화 전에도 안전하도록 기본값 1 제공
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
            # [수정] 상단 여백 제거하여 채팅창 바닥과 높이 정렬
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([4.0, 1.2, 0.4, 0.4])

            with c1:
                # 우측의 넓은 공간을 차지하는 슬라이더
                # [수정] key가 있으므로 value 인자는 제거 (중복 설정 방지)
                new_page = st.slider(
                    "page_nav_wide",
                    min_value=1,
                    max_value=total_pages,
                    key="pdf_nav_slider_wide",
                    disabled=is_generating,
                    label_visibility="collapsed",
                )
                # 슬라이더 조작 시 current_page 동기화
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
                    # 슬라이더 상태도 함께 업데이트
                    st.session_state.pdf_nav_slider_wide = st.session_state.current_page
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
                    # 슬라이더 상태도 함께 업데이트
                    st.session_state.pdf_nav_slider_wide = st.session_state.current_page
                    st.rerun()

    except Exception as e:
        with viewer_container:
            st.error(f"PDF 오류: {e}")


def inject_custom_css():
    """앱 전반에 걸친 최소한의 커스텀 CSS만 주입합니다."""
    st.markdown(
        """
    <style>
    /* 1. 전체 앱 화면 고정 및 전역 스크롤 차단 */
    [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* 2. 메인 영역 및 사이드바 패딩 및 높이 최적화 */
    [data-testid="stMainBlockContainer"] {
        height: 100vh !important;
        padding-top: 2rem !important; /* 상단 여백 통일 */
        padding-bottom: 0rem !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 2rem !important; /* 상단 여백 통일 */
    }

    /* 3. 컨테이너 내부 스크롤 활성화 */
    [data-testid="stVerticalBlockBorderWrapper"] > div:nth-child(1) {
        height: 100% !important;
        overflow-y: auto !important;
    }

    /* 4. JS 측정기 등 커스텀 컴포넌트의 유령 공간 제거 */
    div[data-testid="stHtml"] iframe, 
    div.element-container:has(iframe[title="streamlit_javascript.st_javascript"]) {
        position: absolute !important;
        top: -9999px !important;
        width: 0 !important;
        height: 0 !important;
        visibility: hidden !important;
    }

    /* 5. 상단 서브헤더 및 사이드바 제목 정렬 */
    h3 {
        height: auto !important;
        line-height: 1.5 !important;
        margin-bottom: 1.2rem !important;
        padding-top: 0.2rem !important; /* 상단 여백 소폭 조정 */
        margin-top: 0rem !important;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
        padding-top: 0rem !important;
    }
    
    /* 6. 툴팁 및 인용 배지 스타일 */
    .tooltip {
        position: relative;
        display: inline-block;
        color: #0068c9;
        font-weight: 600;
        cursor: help;
        text-decoration: underline dotted;
    }
    .citation-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #f0f2f6;
        color: #0068c9;
        font-size: 0.75rem;
        font-weight: bold;
        padding: 0 6px;
        margin: 0 2px;
        border-radius: 4px;
        border: 1px solid #d1d5db;
        cursor: default;
        vertical-align: middle;
        height: 1.2rem;
        min-width: 1.2rem;
    }
    .citation-badge:hover {
        background-color: #0068c9;
        color: white;
        border-color: #0068c9;
    }
    /* 사고 과정 컨테이너 */
    .thought-container {
        border-left: 3px solid #ddd;
        padding-left: 15px;
        margin: 10px 0;
        color: #666;
        font-style: italic;
    }
    /* 사이드바 요소 간격 압축 */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def render_sidebar(
    file_uploader_callback: Callable,
    model_selector_callback: Callable,
    embedding_selector_callback: Callable,
    is_generating: bool = False,
    current_file_name: str | None = None,
    current_embedding_model: str | None = None,
    available_models: list[str] | None = None,
):
    with st.sidebar:
        # 1. 브랜드 헤더
        st.title("🤖 GraphRAG")
        st.caption("Local Inference Model")
        st.divider()

        # 2. 문서 관리
        st.subheader("📄 Document")
        st.file_uploader(
            "Upload PDF",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            disabled=is_generating,
            label_visibility="collapsed",
        )
        if current_file_name:
            st.success(f"Active: {current_file_name}")

        st.divider()

        # 3. 모델 설정
        st.subheader("⚙️ Model Settings")
        from common.config import DEFAULT_OLLAMA_MODEL

        if available_models is None:
            st.info("Loading...")
        else:
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

            last_model = SessionManager.get("last_selected_model")
            if not last_model or (actual_models and last_model not in actual_models):
                last_model = (
                    DEFAULT_OLLAMA_MODEL
                    if DEFAULT_OLLAMA_MODEL in actual_models
                    else (actual_models[0] if actual_models else available_models[0])
                )
                SessionManager.set("last_selected_model", last_model)

            st.selectbox(
                "추론 모델 (Inference)",
                available_models,
                index=available_models.index(last_model)
                if last_model in available_models
                else 0,
                key="model_selector",
                on_change=model_selector_callback,
                disabled=is_ollama_error or is_generating,
                label_visibility="collapsed",
            )

        with st.expander("🛠️ Advanced Settings"):
            last_emb = current_embedding_model or AVAILABLE_EMBEDDING_MODELS[0]
            st.selectbox(
                "임베딩 모델 (Embedding)",
                AVAILABLE_EMBEDDING_MODELS,
                index=AVAILABLE_EMBEDDING_MODELS.index(last_emb)
                if last_emb in AVAILABLE_EMBEDDING_MODELS
                else 0,
                key="embedding_model_selector",
                on_change=embedding_selector_callback,
                disabled=is_generating or (available_models is None),
            )


def render_left_column():
    _chat_fragment()


def render_message(
    role: str,
    content: str,
    thought: str | None = None,
    doc_ids: list[Any] | None = None,
    metrics: dict | None = None,
):
    if role == "system":
        with st.chat_message("system", avatar="⚙️"):
            st.caption("시스템 작업 기록")
            st.markdown(content)
        return

    avatar_icon = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar_icon):
        if thought and thought.strip():
            with st.expander("🧠 사고 완료", expanded=False):
                st.markdown(
                    f'<div class="thought-container">{thought}</div>',
                    unsafe_allow_html=True,
                )

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

        # [추가] 성능 메트릭 및 피드백 섹션
        if role == "assistant":
            m_col1, m_col2 = st.columns([0.7, 0.3])

            with m_col2:
                # 고유 키 생성을 위해 내용의 해시 사용
                import hashlib

                msg_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                st.feedback("thumbs", key=f"fb_hist_{msg_hash}")

            with m_col1:
                if metrics:
                    # 표준 캡션 사용
                    st.caption(
                        f"⏱️ {metrics.get('duration', 0):.1f}s | "
                        f"🚀 {metrics.get('tps', 0):.1f} t/s | "
                        f"📄 {metrics.get('doc_count', 0)} refs | "
                        f"🤖 {metrics.get('model', 'Unknown')}"
                    )

        # [추가] 이력 메시지에서도 출처 칩 표시 (참고 문서가 있는 경우)
        if role == "assistant" and documents:
            st.divider()
            
            # 중복 제거 및 페이지 정렬 (이동을 위해 페이지 번호 저장)
            unique_sources = {}
            for doc in documents:
                src = doc.metadata.get("source", "Unknown")
                pg = doc.metadata.get("page", "?")
                key = f"📄 {src} (p.{pg})"
                if key not in unique_sources:
                    unique_sources[key] = pg

            import hashlib
            msg_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            pill_key = f"pills_hist_{msg_hash}"
            
            # [수정] 위젯 생성 전 상태를 먼저 체크하고 처리하여 StreamlitAPIException 방지
            if pill_key in st.session_state and st.session_state[pill_key]:
                selection = st.session_state[pill_key]
                target_pg = unique_sources.get(selection)
                if target_pg and str(target_pg).isdigit():
                    new_pg = int(target_pg)
                    st.session_state.current_page = new_pg
                    # 슬라이더 위젯 상태도 강제 업데이트하여 동기화
                    st.session_state.pdf_nav_slider_wide = new_pg
                    # 상태 초기화 (위젯 생성 전이므로 안전)
                    st.session_state[pill_key] = None
                    st.rerun()

            selected_pill = st.pills(
                "📍 참고 지식 (클릭 시 이동):",
                options=list(unique_sources.keys()),
                selection_mode="single",
                key=pill_key,
            )


def update_window_height():
    """JavaScript를 통해 브라우저 창의 실제 높이를 측정하고 세션 상태에 저장합니다."""
    from streamlit_javascript import st_javascript
    
    # 윈도우 전체 높이 획득 (단 한 번만 호출됨)
    win_h = st_javascript("window.innerHeight", key="height_tracker")
    
    if win_h and win_h > 100:
        st.session_state.last_valid_height = int(win_h)


def _chat_fragment():
    # 1. 이미 계산된 높이 가져오기 (폴백 700)
    win_h = st.session_state.get("last_valid_height", 800)
    container_h = max(400, win_h - 250) # 상하단 여백 제외
    
    chat_container = st.container(height=container_h, border=True)
    # [수정] 세션 초기화 전에도 안전하도록 기본값 [] 제공
    messages = SessionManager.get_messages() or []
    pdf_path = SessionManager.get("pdf_file_path")
    pdf_processed = SessionManager.get("pdf_processed", False)
    is_generating = bool(st.session_state.get("is_generating_answer", False))

    # 문서 분석 중인지 판별 (파일은 업로드됐는데 아직 처리가 안 된 상태)
    is_processing_pdf = bool(pdf_path and not pdf_processed)

    # 1. 채팅 이력 렌더링
    with chat_container:
        system_buffer = []
        insight_rendered = False

        def flush_system_buffer():
            if not system_buffer:
                return

            with st.chat_message("system", avatar="⚙️"):
                log_items = []
                is_ready = False
                has_error = False
                
                chars_to_remove = ["✅", "⏳", "❌", "⚙️", "📄", "ℹ️", "🧠", "✨", "🔄", "⏳", "🎯"]
                
                for m in system_buffer:
                    if m == "READY_FOR_QUERY":
                        is_ready = True
                        continue
                    
                    if "❌" in m or "오류" in m or "실패" in m:
                        has_error = True
                        
                    clean_m = m
                    for char in chars_to_remove:
                        clean_m = clean_m.replace(char, "")
                    
                    clean_m = clean_m.strip()
                    if clean_m:
                        log_items.append(f"└─ {'`ERROR`' if has_error else '`SUCCESS`'} {clean_m}")
                
                # 결과 출력 로직 최적화
                if is_ready and not has_error:
                    # 모두 성공했다면 요약 메시지만 표시
                    st.markdown("**시스템 구성 및 데이터 분석을 완료했습니다.**")
                    st.markdown("\n**이제 문서 내용에 대해 궁금한 점을 질문해 주세요!**")
                else:
                    # 진행 중이거나 에러가 있다면 상세 로그 표시
                    st.markdown("**시스템 작업 기록**\n")
                    st.markdown("  \n".join(log_items))

            system_buffer.clear()

        for msg in messages:
            if msg["role"] == "system":
                system_buffer.append(msg["content"])
            else:
                # 일반 메시지가 나오기 전에 버퍼에 쌓인 시스템 메시지들을 먼저 출력
                flush_system_buffer()
                render_message(
                    msg["role"],
                    msg["content"],
                    thought=msg.get("thought"),
                    doc_ids=msg.get("doc_ids"),
                    metrics=msg.get("metrics"),
                )

        # 반복문 종료 후 남아있는 시스템 메시지 처리
        flush_system_buffer()

        if not messages:
            # 시스템 온보딩 가이드 (⚙️) - 더 간결하게 수정
            with st.chat_message("system", avatar="⚙️"):
                st.caption("🚀 RAG System Quick Start")
                st.markdown("""
                **지능형 문서 분석 모델이 활성화되었습니다.**
                
                1. **문서 업로드**: 사이드바에서 PDF 파일을 업로드하세요.
                2. **심층 질의**: 문서 내용에 기반한 질문을 시작하세요.
                """)
                st.caption("💡 Tip: 답변 하단의 출처 칩을 클릭하여 원문을 확인할 수 있습니다.")

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

    # [추가] 추천 질문 버튼 클릭 처리 및 일반 입력 통합
    user_query = st.chat_input(input_placeholder, disabled=input_disabled, key="chat_input_clean")
    
    # 버튼 클릭 등으로 대기 중인 질문이 있다면 우선 처리
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query # 처리 후 삭제

    if user_query:
        SessionManager.add_message("user", user_query)
        SessionManager.add_status_log("질문 분석 중")

        # UI 즉시 업데이트
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
            final_metrics = result.get("metrics")

            if final_answer and not final_answer.startswith("❌"):
                SessionManager.add_message(
                    "assistant",
                    final_answer,
                    thought=final_thought,
                    documents=final_docs,
                    metrics=final_metrics,
                )
                SessionManager.replace_last_status_log("답변 작성 완료")
                SessionManager.add_status_log("질문 가능")
                st.rerun()
        else:
            with chat_container:
                st.error(f"⚠️ {MSG_CHAT_NO_QA_SYSTEM}")
