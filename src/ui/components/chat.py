"""
채팅 인터페이스 및 스트리밍 응답 관련 컴포넌트.
"""

import logging
import re
import time
from contextlib import aclosing
from typing import Any, TypedDict, cast

import streamlit as st

from api.streaming_handler import get_adaptive_controller, get_streaming_handler
from common.config import (
    MSG_CHAT_GUIDE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_PREPARING_ANSWER,
    MSG_THINKING,
    UI_CONTAINER_HEIGHT,
)
from common.utils import (
    apply_tooltips_to_response,
    format_error_message,
    normalize_latex_delimiters,
    sync_run,
)
from core.session import SessionManager

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    full_response: str
    full_thought: str
    retrieved_docs: list[Any]
    performance: dict[str, Any]
    thinking_start_time: float
    thinking_end_time: float


async def _stream_chat_response(rag_sys, user_query: str, placeholder):
    """RAG 시스템의 스트리밍 응답을 UI에 렌더링 (플레이스홀더 기반 잔상 방지 버전)"""
    state: ChatState = {
        "full_response": "",
        "full_thought": "",
        "retrieved_docs": [],
        "performance": {},
        "thinking_start_time": 0.0,
        "thinking_end_time": 0.0,
    }

    controller = get_adaptive_controller()
    handler = get_streaming_handler()
    model_name = SessionManager.get("last_selected_model")
    session_id = SessionManager.get_session_id()

    last_render_time = time.time()
    render_interval = 0.05

    try:
        # [핵심] 전달받은 플레이스홀더 내부에서만 렌더링하여 위치를 고정
        with placeholder.container():
            with st.chat_message("assistant", avatar="🤖"):
                # 1. 상태창 (key 인자 미지원으로 제거)
                status_container = st.status(MSG_PREPARING_ANSWER, expanded=True)

                # 2. 하단 영역 선언 (이중 렌더링 방지를 위해 empty() 활용)
                thought_area = st.container()
                answer_area = st.empty()

                with status_container:
                    status_log_area = st.container()

                    event_generator = await rag_sys.astream(
                        user_query, model_name=model_name
                    )

                    stream = cast(Any, event_generator)
                    event_stream = (
                        handler.stream_graph_events(
                            stream, adaptive_controller=controller
                        )
                        if handler
                        else stream
                    )

                    async with aclosing(cast(Any, event_stream)) as final_stream:
                        async for chunk in final_stream:
                            if chunk.status:
                                status_container.update(
                                    label=f"⏳ {chunk.status}", state="running"
                                )
                                with status_log_area:
                                    st.caption(f"▹ {chunk.status}")
                                SessionManager.add_status_log(chunk.status)

                            if chunk.metadata and "documents" in chunk.metadata:
                                state["retrieved_docs"] = chunk.metadata["documents"]

                            if chunk.performance:
                                state["performance"] = chunk.performance

                            if chunk.thought:
                                if not state["full_thought"]:
                                    state["thinking_start_time"] = time.time()
                                    with thought_area:
                                        thought_display = st.empty()
                                state["full_thought"] += chunk.thought
                                if time.time() - last_render_time > render_interval:
                                    thought_display.markdown(
                                        f"*{state['full_thought']}*"
                                    )
                                    last_render_time = time.time()

                            if chunk.content:
                                if not state["full_response"]:
                                    status_container.update(
                                        label="✅ 분석 완료 및 답변 작성 중",
                                        state="complete",
                                        expanded=False,
                                    )
                                    state["thinking_end_time"] = time.time()
                                    if (
                                        state["full_thought"]
                                        and "thought_display" in locals()
                                    ):
                                        thought_display.empty()
                                        with thought_area:
                                            dur = (
                                                state["thinking_end_time"]
                                                - state["thinking_start_time"]
                                            )
                                            with st.expander(
                                                f"{MSG_THINKING[:-3]} ({dur:.1f}초)",
                                                expanded=False,
                                            ):
                                                st.markdown(
                                                    f'<div class="thought-container">{state["full_thought"]}</div>',
                                                    unsafe_allow_html=True,
                                                )

                                state["full_response"] += chunk.content
                                if (
                                    time.time() - last_render_time > render_interval
                                    or chunk.is_final
                                ):
                                    display_text = _clean_response_redundancy(
                                        state["full_response"]
                                    )
                                    display_text = normalize_latex_delimiters(
                                        display_text
                                    )
                                    if state["retrieved_docs"]:
                                        display_text = apply_tooltips_to_response(
                                            display_text, state["retrieved_docs"]
                                        )

                                    cursor = "▌" if not chunk.is_final else ""
                                    answer_area.markdown(
                                        display_text + cursor, unsafe_allow_html=True
                                    )
                                    last_render_time = time.time()

                # 최종 상태 업데이트
                status_container.update(
                    label="✨ 답변 생성 완료", state="complete", expanded=False
                )

            # [수정] 결과 반환 전 PDF 하이라이트 등 후처리 수행
            if state["retrieved_docs"]:
                from common.utils import extract_annotations_from_docs

                annotations = extract_annotations_from_docs(state["retrieved_docs"])
                SessionManager.set("pdf_annotations", annotations)
                try:
                    first_doc = state["retrieved_docs"][0]
                    meta = (
                        getattr(first_doc, "metadata", {})
                        if hasattr(first_doc, "metadata")
                        else first_doc.get("metadata", {})
                    )
                    target_p = meta.get("page")
                    if target_p is not None:
                        SessionManager.set("pdf_target_page", int(target_p))
                except Exception as e:
                    logger.warning(f"자동 점프 페이지 추출 실패: {e}")

            return {
                "response": state["full_response"],
                "processed_content": apply_tooltips_to_response(
                    _clean_response_redundancy(state["full_response"]),
                    state["retrieved_docs"],
                ),
                "thought": state["full_thought"],
                "documents": state["retrieved_docs"],
                "performance": state["performance"],
            }
    except Exception as e:
        logger.error(f"UI 스트리밍 오류: {e}", exc_info=True)
        return {"response": format_error_message(e), "thought": "", "documents": []}
    finally:
        SessionManager.set("is_generating_answer", False)
        # [핵심] 스트리밍용 임시 UI 제거 (rerun 시 히스토리에서 정식 출력됨)
        placeholder.empty()


def _clean_response_redundancy(text: str) -> str:
    if not text:
        return text
    clean_patterns = [
        r"^#{1,4}\s*(?:답변|결과|분석 결과|Response|Answer|Result)[:\s]*",
        r"^\**\s*(?:답변|결과|분석 결과|Response|Answer|Result)[:\s]*\**\s*",
    ]
    result = text.strip()
    for pattern in clean_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE).strip()
    return result


def render_message(
    role: str,
    content: str,
    thought: str | None = None,
    documents: list[Any] | None = None,
    metrics: dict | None = None,
    processed_content: str | None = None,
    msg_type: str = "general",
    wrap_in_container: bool = True,
    msg_index: int = 0,
    **kwargs,
):
    """메시지를 렌더링하는 통합 엔진."""
    avatar_icon = "🤖" if role == "assistant" else "👤"
    msg_id = kwargs.get("msg_id", f"msg_{msg_index}")

    with (
        st.chat_message(role, avatar=avatar_icon)
        if wrap_in_container
        else st.container()
    ):
        if thought and thought.strip():
            with st.expander(
                MSG_THINKING[:-3] + " 완료", expanded=False, key=f"exp_{msg_id}"
            ):
                st.markdown(
                    f'<div class="thought-container" style="font-size: 0.85rem;">{thought}</div>',
                    unsafe_allow_html=True,
                )

        if processed_content:
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            display_text = normalize_latex_delimiters(content)
            if role == "assistant" and documents:
                display_text = apply_tooltips_to_response(display_text, documents)
            st.markdown(display_text, unsafe_allow_html=True)

        if role == "assistant" and metrics:
            st.divider()
            m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1.2, 2.2])
            with m_col1:
                st.markdown(
                    f"📥 **{metrics.get('input_token_count', 0)}** <small>In</small>",
                    help="입력 토큰 수",
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"📤 **{metrics.get('token_count', metrics.get('output_token_count', 0))}** <small>Out</small>",
                    help="출력 토큰 수",
                    unsafe_allow_html=True,
                )
            with m_col3:
                if documents:
                    # 안전하게 페이지 번호 추출 (Document 객체와 딕셔너리 혼용 지원)
                    extracted_pages = set()
                    for d in documents:
                        meta = (
                            getattr(d, "metadata", d)
                            if hasattr(d, "metadata")
                            else d.get("metadata", {})
                        )
                        p = meta.get("page", 1)
                        try:
                            extracted_pages.add(int(p))
                        except (ValueError, TypeError):
                            extracted_pages.add(1)

                    pages = sorted(extracted_pages)

                    with st.popover(
                        f"📄 **{len(documents)}** Docs", use_container_width=True
                    ):
                        cols = st.columns(min(len(pages), 3))
                        for idx, p in enumerate(pages):
                            if cols[idx % 3].button(
                                f"{p}p", key=f"jump_{msg_id}_{p}_{idx}"
                            ):
                                SessionManager.set("pdf_target_page", p)
                                st.rerun()

                else:
                    st.markdown("📄 **0** <small>Docs</small>", unsafe_allow_html=True)
            with m_col4:
                total = metrics.get("total_time", 0)
                tps = metrics.get("tps", metrics.get("tokens_per_second", 0))
                st.markdown(
                    f"⏱️ **{total:.1f}s** <small>({tps:.1f}t/s)</small>",
                    help=f"총 {total:.2f}초",
                    unsafe_allow_html=True,
                )


def _render_system_logs(logs: list[str]):
    """시스템 로그를 익스팬더로 그룹화하여 렌더링"""
    if not logs:
        return
    latest_log = logs[-1].strip()
    if len(latest_log) > 35:
        latest_log = latest_log[:32] + "..."

    with (
        st.chat_message("system", avatar="⚙️"),
        st.expander(f"⚙️ {latest_log}", expanded=False),
    ):
        for log in logs:
            st.caption(f"▹ {log}")


@st.fragment
def render_chat_interface():
    """채팅 인터페이스 (Fragment 격리 및 잔상 방지)"""
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=False)
    messages = SessionManager.get_messages() or []
    is_generating = bool(SessionManager.get("is_generating_answer", False))
    current_sid = SessionManager.get_session_id()

    with chat_container:
        if not messages:
            st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)

        current_log_group = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" or msg.get("msg_type") == "log":
                if content != "READY_FOR_QUERY":
                    current_log_group.append(content)
            else:
                if current_log_group:
                    _render_system_logs(current_log_group)
                    current_log_group = []
                render_message(
                    role=role,
                    content=content,
                    thought=msg.get("thought"),
                    documents=msg.get("documents"),
                    metrics=msg.get("metrics"),
                    processed_content=msg.get("processed_content"),
                    msg_index=i,
                    msg_id=msg.get("msg_id"),
                )

        if current_log_group:
            _render_system_logs(current_log_group)

        # [핵심] 스트리밍 전용 플레이스홀더를 히스토리 렌더링 직후에 배치
        streaming_placeholder = st.empty()

    user_query = st.chat_input(MSG_CHAT_INPUT_PLACEHOLDER, disabled=is_generating)

    if user_query and not is_generating:
        SessionManager.add_message("user", user_query)
        try:
            st.rerun(scope="fragment")
        except Exception:
            st.rerun()

    # 대기 중인 질문이 있고 아직 답변 생성 전인 경우
    if not is_generating and messages and messages[-1].get("role") == "user":
        from core.rag_core import RAGSystem

        rag_sys = RAGSystem(session_id=current_sid)

        if SessionManager.is_ready_for_chat(session_id=current_sid):
            SessionManager.set("is_generating_answer", True)
            # 스트리밍 실행 (플레이스홀더 전달)
            result = sync_run(
                _stream_chat_response(
                    rag_sys, messages[-1]["content"], streaming_placeholder
                )
            )

            if result and result.get("response"):
                SessionManager.add_message(
                    role="assistant",
                    content=result["response"],
                    thought=result.get("thought"),
                    documents=result.get("documents"),
                    metrics=result.get("performance"),
                    processed_content=result.get("processed_content"),
                )
            try:
                st.rerun(scope="fragment")
            except Exception:
                st.rerun()
