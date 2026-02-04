"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Integrated Messaging Version: 시스템 로그와 일반 메시지의 일관된 관리 및 렌더링.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import aclosing
from typing import Any

import streamlit as st

from api.streaming_handler import get_adaptive_controller, get_streaming_handler
from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    MSG_CHAT_GUIDE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_PDF_VIEWER_NO_FILE,
)
from common.utils import (
    apply_tooltips_to_response,
    format_error_message,
    normalize_latex_delimiters,
    sync_run,
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
        "annotations": [],
        "performance": None,
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

    handler = get_streaming_handler()
    controller = get_adaptive_controller()

    last_render_time = 0.0
    render_interval = 0.05

    try:
        with chat_container, st.chat_message("assistant", avatar="🤖"):
            # 진행 상태 로그용 컨테이너 (st.status 활용)
            status_box = st.status("🚀 파이프라인 가동 중...", expanded=True)

            def update_status(msg: str, state="running"):
                status_box.write(f"└─ {msg}")
                if state == "complete":
                    status_box.update(
                        label="✅ 분석 완료", state="complete", expanded=False
                    )

            thought_area = st.container()
            answer_area = st.empty()

            event_generator = rag_engine.astream_events(
                {"input": user_query}, config=run_config, version="v2"
            )

            async with aclosing(
                handler.stream_graph_events(
                    event_generator, adaptive_controller=controller
                )
            ) as stream:
                async for chunk in stream:
                    if chunk.status:
                        update_status(chunk.status)
                        SessionManager.add_status_log(chunk.status)

                    if chunk.metadata:
                        if "documents" in chunk.metadata:
                            state["retrieved_docs"] = chunk.metadata["documents"]
                            update_status(
                                f"관련 지식 {len(state['retrieved_docs'])}개 확보"
                            )
                        if "annotations" in chunk.metadata:
                            st.session_state.pdf_annotations = chunk.metadata[
                                "annotations"
                            ]
                            state["annotations"] = chunk.metadata["annotations"]

                    if chunk.performance:
                        state["performance"] = chunk.performance

                    if chunk.thought:
                        if not state["full_thought"]:
                            state["thinking_start_time"] = time.time()
                            with thought_area:
                                st.caption("AI의 사고 흐름:")
                                thought_display = st.empty()
                        state["full_thought"] += chunk.thought
                        if time.time() - last_render_time > render_interval:
                            thought_display.markdown(f"*{state['full_thought']}*")
                            last_render_time = time.time()

                    if chunk.content:
                        if not state["full_response"]:
                            update_status("답변 생성 중...", state="complete")
                            state["thinking_end_time"] = time.time()
                            if state["full_thought"]:
                                with thought_area:
                                    thinking_dur = (
                                        state["thinking_end_time"]
                                        - state["thinking_start_time"]
                                    )
                                    with st.expander(
                                        f"💭 사고 완료 ({thinking_dur:.1f}초)",
                                        expanded=False,
                                    ):
                                        st.markdown(state["full_thought"])
                                    if "thought_display" in locals():
                                        thought_display.empty()

                        state["full_response"] += chunk.content
                        if (
                            time.time() - last_render_time > render_interval
                            or chunk.is_final
                        ):
                            # 마지막에는 커서 제거
                            cursor = "▌" if not chunk.is_final else ""
                            answer_area.markdown(state["full_response"] + cursor)
                            last_render_time = time.time()

            _finalize_ui_rendering(thought_area, answer_area, state)

            return {
                "response": state["full_response"],
                "thought": state["full_thought"],
                "documents": state["retrieved_docs"],
                "performance": state["performance"],
            }

    except Exception as e:
        logger.error(f"UI 스트리밍 오류: {e}", exc_info=True)
        friendly_msg = format_error_message(e)
        SessionManager.add_status_log(friendly_msg)
        return {"response": friendly_msg, "thought": "", "documents": []}
    finally:
        SessionManager.set("is_generating_answer", False)


def _finalize_ui_rendering(thought_container, answer_display, state):
    """답변 생성이 끝난 후 UI를 최종 상태로 정리합니다."""
    if state["full_response"]:
        processed = normalize_latex_delimiters(state["full_response"])
        if state["retrieved_docs"]:
            processed = apply_tooltips_to_response(processed, state["retrieved_docs"])

        answer_display.empty()
        with answer_display:
            render_message(
                role="assistant",
                content=state["full_response"],
                thought=state["full_thought"],
                documents=state["retrieved_docs"],
                metrics=state["performance"],
                processed_content=processed,
                annotations=state.get("annotations"),
            )
    else:
        answer_display.error("⚠️ 답변이 생성되지 않았습니다.")


def render_message(
    role: str,
    content: str,
    thought: str | None = None,
    doc_ids: list[str] | None = None,
    documents: list[Any] | None = None,
    metrics: dict | None = None,
    processed_content: str | None = None,
    msg_type: str = "general",
    annotations: list[dict] | None = None,
    source_file: str | None = None,
    **kwargs,
):
    """메시지를 렌더링하는 통합 엔진. msg_type에 따라 레이아웃 자동 결정."""

    if role == "system" or msg_type == "log":
        with st.chat_message("system", avatar="⚙️"):
            if "완료" in content or "성공" in content:
                st.success(content, icon="✅")
            elif "실패" in content or "오류" in content:
                st.error(content, icon="❌")
            else:
                st.info(content, icon="ℹ️")
        return

    avatar_icon = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar_icon):
        # 1. 사고 과정 (Thought)
        if thought and thought.strip():
            with st.expander("🧠 사고 완료", expanded=False):
                st.markdown(
                    f'<div class="thought-container">{thought}</div>',
                    unsafe_allow_html=True,
                )

        # 2. 답변 본문 (Content)
        if processed_content:
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            display_text = normalize_latex_delimiters(content)
            if not documents and doc_ids:
                doc_pool = SessionManager.get("doc_pool", {})
                documents = [doc_pool[d_id] for d_id in doc_ids if d_id in doc_pool]
            if documents:
                display_text = apply_tooltips_to_response(display_text, documents)
            st.markdown(display_text, unsafe_allow_html=True)

        # 3. 어시스턴트 전용 추가 UI (하단 배치)
        if role == "assistant":
            # [최적화] 현재 문서와 메시지 출처가 일치할 때만 바로가기 표시
            active_file = SessionManager.get("last_uploaded_file_name")
            msg_source = source_file or (
                annotations[0].get("source") if annotations else None
            )

            jump_pages: list[int] = []
            # 출처가 다르거나 없으면 바로가기 생략 (모순 방지)
            if not (active_file and msg_source and active_file != msg_source):
                current_annos = annotations or st.session_state.get(
                    "pdf_annotations", []
                )
                if current_annos:
                    jump_pages = sorted({h["page"] for h in current_annos})
                elif documents:
                    jump_pages = sorted(
                        {doc.metadata.get("page", 1) for doc in documents}
                    )

            if jump_pages:
                st.markdown(
                    "<div style='margin-top: 10px;'></div>", unsafe_allow_html=True
                )
                st.caption("📍 관련 페이지 바로가기")

                msg_hash = hashlib.md5(
                    content.encode(), usedforsecurity=False
                ).hexdigest()[:8]
                cols = st.columns(min(len(jump_pages), 8) + 1)

                # [최적화] 명시적 타입 캐스팅을 통해 Mypy 오류 해결
                from typing import cast

                final_jump_pages = cast(list[int], jump_pages)
                for i, p_num in enumerate(final_jump_pages):
                    if i < len(cols) and cols[i].button(
                        f"{p_num}p",
                        key=f"jmp_{msg_hash}_{p_num}_{i}",
                        use_container_width=True,
                    ):
                        # [최적화] 현재 문서의 페이지 범위 확인
                        pdf_path = SessionManager.get("pdf_file_path")
                        if pdf_path and os.path.exists(pdf_path):
                            import fitz

                            with fitz.open(pdf_path) as doc:
                                max_p = len(doc)
                            if p_num > max_p:
                                st.toast(
                                    f"현재 문서에는 {p_num}페이지가 없습니다.",
                                    icon="⚠️",
                                )
                                return

                        st.session_state.pdf_page_index = p_num

                        if current_annos:
                            st.session_state.pdf_annotations = current_annos
                            for h in current_annos:
                                if h.get("page") == p_num:
                                    st.session_state.active_ref_id = h.get("id")
                                    break

                            # [최적화] 명시적 타입 캐스팅을 통해 Mypy 오류 해결
                            final_annos = cast(list[dict], current_annos)
                            for idx_h, h in enumerate(final_annos):
                                if h.get("page") == p_num:
                                    st.session_state.scroll_target_idx = idx_h + 1
                                    break
                        st.rerun()

            st.divider()
            m_col1, m_col2 = st.columns([0.75, 0.25])
            with m_col2:
                m_hash = hashlib.md5(
                    content.encode(), usedforsecurity=False
                ).hexdigest()[:8]
                st.feedback("thumbs", key=f"fb_{m_hash}")
            with m_col1:
                if metrics:
                    st.caption(
                        f"⏱️ {metrics.get('total_time', 0):.1f}s (TTFT: {metrics.get('ttft', 0):.2f}s) | "
                        f"🚀 {metrics.get('tps', 0):.1f} t/s | 🤖 {metrics.get('model_name', 'Unknown')}"
                    )


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.cache_resource(show_spinner=False)
def _get_pdf_info(pdf_path: str) -> tuple[int, bytes]:
    import fitz

    if not os.path.exists(pdf_path):
        return 0, b""
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return total_pages, pdf_bytes


@st.fragment
def _pdf_viewer_fragment():
    from streamlit_pdf_viewer import pdf_viewer

    win_h = st.session_state.get("last_valid_height", 800)
    viewer_h = max(400, win_h - 250)

    viewer_container = st.container(height=viewer_h, border=True)

    pdf_path_raw = SessionManager.get("pdf_file_path", None)
    if not pdf_path_raw:
        with viewer_container:
            st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    try:
        total_pages, pdf_bytes = _get_pdf_info(pdf_path)
        if total_pages == 0:
            with viewer_container:
                st.error("⚠️ PDF 로드 실패")
            return

        if "pdf_page_index" not in st.session_state:
            st.session_state.pdf_page_index = SessionManager.get("current_page", 1)

        if "pdf_render_text" not in st.session_state:
            st.session_state.pdf_render_text = True

        raw_highlights = st.session_state.get("pdf_annotations", [])
        highlights = []
        active_id = st.session_state.get("active_ref_id")
        if isinstance(raw_highlights, list):
            for h in raw_highlights:
                if isinstance(h, dict) and all(
                    k in h for k in ["page", "x", "y", "width", "height"]
                ):
                    processed_h = h.copy()
                    if active_id and h.get("id") == active_id:
                        processed_h["color"] = "rgba(255, 0, 0, 0.5)"
                        processed_h["border"] = "solid"
                    else:
                        processed_h["color"] = "rgba(255, 0, 0, 0.2)"
                    highlights.append(processed_h)

        viewer_params = {
            "input": pdf_bytes,
            "pages_to_render": [st.session_state.pdf_page_index],
            "render_text": st.session_state.get("pdf_render_text", True),
            "annotation_outline_size": 2,
        }
        if highlights:
            viewer_params["annotations"] = highlights
            if s_idx := st.session_state.get("scroll_target_idx"):
                viewer_params["scroll_to_annotation"] = s_idx
                del st.session_state.scroll_target_idx

        with viewer_container:
            pdf_viewer(**viewer_params)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        # [최적화] 컨트롤바 우측 정렬을 위해 컬럼 순서 및 비율 조정
        c_spacer, c_page, c_set = st.columns([3.0, 1.5, 0.5])

        with c_page:
            sub_col1, sub_col2 = st.columns([1, 1])
            with sub_col1:
                # [최적화] 현재 페이지 번호가 새 문서의 전체 페이지 수를 넘지 않도록 보정
                safe_page_idx = min(
                    max(1, st.session_state.pdf_page_index), total_pages
                )
                st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    key="pdf_page_index_input",
                    value=safe_page_idx,
                    on_change=lambda: SessionManager.set(
                        "current_page", st.session_state.pdf_page_index_input
                    ),
                    label_visibility="collapsed",
                )
                # 상태 동기화
                st.session_state.pdf_page_index = st.session_state.pdf_page_index_input
            with sub_col2:
                st.markdown(
                    f"<div style='line-height: 2.3rem; white-space: nowrap;'>/ {total_pages} p</div>",
                    unsafe_allow_html=True,
                )

        with c_set, st.popover("⚙️", use_container_width=True):
            st.caption("📝 텍스트 설정")
            st.session_state.pdf_render_text = st.toggle(
                "텍스트 선택 가능", value=st.session_state.pdf_render_text
            )

    except Exception as e:
        st.error(f"PDF 오류: {e}")


def inject_custom_css():
    """앱 전반에 걸친 최소한의 커스텀 CSS 및 JS 주입."""
    st.markdown(
        """
    <style>
    [data-testid="stAppViewContainer"] { height: 100vh !important; overflow: hidden !important; }
    [data-testid="stMainBlockContainer"] { height: 100vh !important; padding-top: 2rem !important; padding-bottom: 0rem !important; overflow: hidden !important; }
    [data-testid="stSidebarContent"] { padding-top: 2rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div:nth-child(1) { height: 100% !important; overflow-y: auto !important; }
    div.element-container:has(iframe[title="streamlit_javascript.st_javascript"]) { display: none !important; }
    .thought-container { font-style: italic; color: #666; border-left: 3px solid #eee; padding-left: 10px; margin-bottom: 10px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating=False,
    current_file_name=None,
    current_embedding_model=None,
    available_models=None,
):
    with st.sidebar:
        st.header("🤖 GraphRAG")

        with st.container(border=True):
            st.subheader("📄 문서 처리")
            st.file_uploader(
                "Upload PDF",
                type="pdf",
                key="pdf_uploader",
                on_change=file_uploader_callback,
                disabled=is_generating,
                label_visibility="collapsed",
            )
            if current_file_name:
                st.caption(f"Active: :green[{current_file_name}]")

        with st.container(border=True):
            st.subheader("🧠 추론 모델")
            if available_models is None:
                st.info("Loading models...")
            else:
                actual_models = [m for m in available_models if "---" not in m]
                last_model = SessionManager.get("last_selected_model")
                if not last_model or (
                    actual_models and last_model not in actual_models
                ):
                    from common.config import DEFAULT_OLLAMA_MODEL

                    last_model = (
                        DEFAULT_OLLAMA_MODEL
                        if DEFAULT_OLLAMA_MODEL in actual_models
                        else (
                            actual_models[0] if actual_models else available_models[0]
                        )
                    )
                    SessionManager.set("last_selected_model", last_model)

                if len(actual_models) <= 4:
                    st.segmented_control(
                        "Inference Model",
                        actual_models,
                        default=last_model,
                        key="model_selector",
                        on_change=model_selector_callback,
                        disabled=is_generating,
                        label_visibility="collapsed",
                    )
                else:
                    try:
                        def_idx = actual_models.index(last_model)
                    except ValueError:
                        def_idx = 0
                    st.selectbox(
                        "Inference Model",
                        actual_models,
                        index=def_idx,
                        key="model_selector",
                        on_change=model_selector_callback,
                        disabled=is_generating,
                        label_visibility="collapsed",
                    )

        with st.container(border=True):
            st.subheader("🛠️ 시스템 상세 설정")
            with st.expander("상세 옵션 보기", expanded=False):
                st.caption("Embedding Model")
                st.pills(
                    "Embedding Model",
                    AVAILABLE_EMBEDDING_MODELS,
                    key="embedding_model_selector",
                    on_change=embedding_selector_callback,
                    disabled=is_generating or (available_models is None),
                    label_visibility="collapsed",
                )
                st.markdown(
                    "<div style='margin-top: 10px;'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "🗑️ 캐시 및 세션 초기화", use_container_width=True, type="secondary"
                ):
                    SessionManager.reset_all_state()
                    st.rerun()


def render_left_column():
    _chat_fragment()


def update_window_height():
    from streamlit_javascript import st_javascript

    win_h = st_javascript("window.innerHeight", key="height_tracker")
    if win_h and win_h > 100:
        st.session_state.last_valid_height = int(win_h)


def _chat_fragment():
    win_h = st.session_state.get("last_valid_height", 800)
    container_h = max(400, win_h - 250)
    chat_container = st.container(height=container_h, border=True)

    messages = SessionManager.get_messages() or []
    pdf_path = SessionManager.get("pdf_file_path")
    pdf_processed = SessionManager.get("pdf_processed", False)
    pdf_error = SessionManager.get("pdf_processing_error")

    is_generating = bool(SessionManager.get("is_generating_answer", False))
    is_processing_pdf = bool(pdf_path and not pdf_processed and not pdf_error)

    with chat_container:
        if not messages:
            st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)

        system_buffer = []

        def flush_system_buffer():
            if not system_buffer:
                return
            with st.chat_message("system", avatar="⚙️"):
                is_ready, has_error = False, False
                log_items = []
                for m in system_buffer:
                    if m == "READY_FOR_QUERY":
                        is_ready = True
                        continue
                    if any(x in m for x in ["❌", "오류", "실패"]):
                        has_error = True
                    log_items.append(f"└─ {m}")
                if is_ready and not has_error:
                    st.markdown("**시스템 구성 및 데이터 분석 완료**")
                    st.markdown("문서 내용에 대해 질문해 주세요!")
                else:
                    st.markdown("  \n".join(log_items))
            system_buffer.clear()

        for msg in messages:
            if msg.get("role") == "system" or msg.get("msg_type") == "log":
                system_buffer.append(msg["content"])
            else:
                flush_system_buffer()
                render_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    thought=msg.get("thought"),
                    doc_ids=msg.get("doc_ids"),
                    metrics=msg.get("metrics"),
                    processed_content=msg.get("processed_content"),
                    msg_type=msg.get("msg_type", "general"),
                    annotations=msg.get("annotations"),
                    source_file=msg.get("source_file"),
                )
        flush_system_buffer()

    input_disabled = is_generating or is_processing_pdf
    input_placeholder = (
        "문서 분석 중..."
        if is_processing_pdf
        else ("답변 생성 중..." if is_generating else MSG_CHAT_INPUT_PLACEHOLDER)
    )

    user_query = st.chat_input(
        input_placeholder, disabled=input_disabled, key="chat_input_clean"
    )

    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query

    if user_query:
        SessionManager.add_message("user", user_query)
        if "last_processed_highlights" in st.session_state:
            del st.session_state.last_processed_highlights
        st.session_state.pdf_annotations = []

        with chat_container:
            render_message("user", user_query)

        rag_engine = SessionManager.get("rag_engine")
        if rag_engine:
            result = sync_run(
                _stream_chat_response(rag_engine, user_query, chat_container)
            )

            final_answer = result.get("response", "")
            final_thought = result.get("thought", "")
            final_docs = result.get("documents", [])
            final_metrics = result.get("performance")

            if final_answer and not final_answer.startswith("❌"):
                processed = normalize_latex_delimiters(final_answer)
                if final_docs:
                    processed = apply_tooltips_to_response(processed, final_docs)

                SessionManager.add_message(
                    role="assistant",
                    content=final_answer,
                    processed_content=processed,
                    thought=final_thought,
                    doc_ids=[
                        hashlib.md5(
                            d.page_content.encode(), usedforsecurity=False
                        ).hexdigest()[:8]
                        for d in final_docs
                    ]
                    if final_docs
                    else [],
                    annotations=st.session_state.get("pdf_annotations", []),
                    metrics=final_metrics,
                    msg_type="answer",
                    source_file=SessionManager.get(
                        "last_uploaded_file_name"
                    ),  # [추가] 출처 문서 기록
                )
                st.rerun()
        else:
            st.error(MSG_CHAT_NO_QA_SYSTEM)
