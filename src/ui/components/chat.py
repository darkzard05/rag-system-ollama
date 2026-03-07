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


async def _stream_chat_response(rag_sys, user_query: str, chat_container):
    """RAG 시스템의 스트리밍 응답을 UI에 렌더링 (Native UI 개편 버전)"""
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

    last_render_time = time.time()
    render_interval = 0.05

    try:
        with st.chat_message("assistant", avatar="🤖"):
            # 1. 상태창(Status Expander)을 먼저 선언하여 상단에 배치
            status_container = st.status(MSG_PREPARING_ANSWER, expanded=True)

            # 2. 그 하단에 사고 과정과 답변이 생성될 영역을 선언
            thought_area = st.container()
            answer_area = st.empty()

            # 3. 상태창 컨텍스트 내부에서 루프를 실행하여 안정성 확보
            with status_container:
                status_log_area = st.container()

                # [복구] astream_events는 async def이므로 await를 사용하여 비동기 제너레이터를 획득해야 함
                event_generator = await rag_sys.astream_events(
                    user_query, model_name=model_name
                )

                # 테스트 환경을 위해 handler를 거치지 않는 경로 지원
                stream = cast(Any, event_generator)
                
                # 만약 event_generator가 dict를 반환한다면 handler가 필요함
                # 하지만 테스트에서는 직접 StreamChunk를 주입할 수 있음
                if handler:
                    event_stream = handler.stream_graph_events(
                        stream, adaptive_controller=controller
                    )
                else:
                    event_stream = stream

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
                                    st.caption("AI Thinking...")
                                    thought_display = st.empty()
                            state["full_thought"] += chunk.thought
                            if time.time() - last_render_time > render_interval:
                                thought_display.markdown(f"*{state['full_thought']}*")
                                last_render_time = time.time()

                        if chunk.content:
                            if not state["full_response"]:
                                # 답변 시작 시 상태창을 접음
                                status_container.update(
                                    label="✅ 분석 완료 및 답변 작성 중",
                                    state="complete",
                                    expanded=False,
                                )
                                state["thinking_end_time"] = time.time()
                                if state["full_thought"]:
                                    with thought_area:
                                        dur = (
                                            state["thinking_end_time"]
                                            - state["thinking_start_time"]
                                        )
                                        with st.expander(
                                            f"🧠 사고 과정 ({dur:.1f}초)",
                                            expanded=False,
                                        ):
                                            st.markdown(
                                                f'<div class="thought-container">{state["full_thought"]}</div>',
                                                unsafe_allow_html=True,
                                            )
                                        if "thought_display" in locals():
                                            thought_display.empty()

                            state["full_response"] += chunk.content
                            if (
                                time.time() - last_render_time > render_interval
                                or chunk.is_final
                            ):
                                display_text = _clean_response_redundancy(
                                    state["full_response"]
                                )
                                display_text = normalize_latex_delimiters(display_text)

                                # [수정] 스트리밍 중에도 하이라이트(툴팁) 적용
                                if state["retrieved_docs"]:
                                    display_text = apply_tooltips_to_response(
                                        display_text, state["retrieved_docs"]
                                    )

                                cursor = "▌" if not chunk.is_final else ""
                                # 하이라이트(HTML) 표현을 위해 unsafe_allow_html=True 추가
                                answer_area.markdown(
                                    display_text + cursor, unsafe_allow_html=True
                                )
                                last_render_time = time.time()

            # 루프 종료 후 최종 업데이트
            status_container.update(
                label="✨ 답변 생성 완료", state="complete", expanded=False
            )
            SessionManager.add_status_log("✨ 답변 생성 완료")

            cleaned_final = _clean_response_redundancy(state["full_response"])
            processed_final = apply_tooltips_to_response(
                cleaned_final, state["retrieved_docs"], msg_index=999
            )  # 임시 인덱스, 실제 렌더링 시 보정됨

            # [핵심 추가] 답변 생성에 사용된 문서들을 기반으로 PDF 하이라이트 생성
            if state["retrieved_docs"]:
                from common.utils import extract_annotations_from_docs

                annotations = extract_annotations_from_docs(state["retrieved_docs"])
                SessionManager.set("pdf_annotations", annotations)

                # [추가] 가장 관련성 높은 첫 번째 페이지로 자동 이동
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
                "processed_content": processed_final,
                "thought": state["full_thought"],
                "documents": state["retrieved_docs"],
                "performance": state["performance"],
            }
    except Exception as e:
        logger.error(f"UI 스트리밍 오류: {e}", exc_info=True)
        friendly_msg = format_error_message(e)
        return {"response": friendly_msg, "thought": "", "documents": []}
    finally:
        SessionManager.set("is_generating_answer", False)


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

    with (
        st.chat_message(role, avatar=avatar_icon)
        if wrap_in_container
        else st.container()
    ):
        if thought and thought.strip():
            with st.expander("🧠 사고 완료", expanded=False):
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

        if role == "assistant":
            st.divider()
            if metrics:
                # [리팩토링] 지표를 4개의 컬럼으로 확장하여 상세 정보 표시
                m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1.2, 1.8])

                with m_col1:
                    in_tokens = metrics.get("input_token_count", 0)
                    st.markdown(
                        f"📥 **{in_tokens}** <small>In</small>", unsafe_allow_html=True
                    )

                with m_col2:
                    out_tokens = metrics.get("token_count", 0)
                    st.markdown(
                        f"📤 **{out_tokens}** <small>Out</small>",
                        unsafe_allow_html=True,
                    )

                with m_col3:
                    doc_count = len(documents) if documents else 0
                    if documents:
                        pages = sorted(
                            {
                                int(d.metadata.get("page", 1))
                                if hasattr(d, "metadata")
                                else int(d.get("metadata", {}).get("page", 1))
                                for d in documents
                            }
                        )
                        if pages:
                            with st.popover(
                                f"📄 {doc_count} Docs", use_container_width=True
                            ):
                                st.caption("근거 페이지로 이동:")
                                cols = st.columns(min(len(pages), 3))
                                for idx, p in enumerate(pages):
                                    if cols[idx % 3].button(
                                        f"{p}p", key=f"jump_{msg_index}_{p}_{idx}"
                                    ):
                                        SessionManager.set("pdf_target_page", p)
                                        st.rerun()
                        else:
                            st.markdown(
                                f"📄 **{doc_count}** <small>Docs</small>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "📄 **0** <small>Docs</small>", unsafe_allow_html=True
                        )

                with m_col4:
                    total = metrics.get("total_time", 0)
                    ttft = metrics.get("first_token_latency", metrics.get("ttft", 0))
                    tps = metrics.get("tokens_per_second", metrics.get("tps", 0))

                    # 지연 시간과 속도를 조합하여 전문적으로 표시
                    st.markdown(
                        f"⏱️ **{total:.1f}s** <small>(🎯{ttft * 1000:.0f}ms / ⚡{tps:.1f}t/s)</small>",
                        unsafe_allow_html=True,
                    )


def _render_system_logs(logs: list[str]):
    """시스템 로그를 익스팬더로 그룹화하여 렌더링 (마지막 과정을 제목에 반영)"""
    if not logs:
        return

    # 1. 프로세스 성격 및 상태 판별
    log_text = "".join(logs)
    is_doc_analysis = any(
        kw in log_text for kw in ["마크다운", "청킹", "인덱싱", "벡터화"]
    )
    is_complete = any("완료" in log_entry or "성공" in log_entry for log_entry in logs)

    # 2. 마지막 로그 추출 (제목용)
    latest_log = logs[-1].strip()
    # 너무 긴 제목 방지
    if len(latest_log) > 35:
        latest_log = latest_log[:32] + "..."

    # 3. 아이콘 및 타이틀 구성
    if is_complete:
        icon = "✅"
        title = f"{icon} {latest_log}"
    else:
        icon = "📄" if is_doc_analysis else "⏳"
        title = f"{icon} {latest_log}"

    with (
        st.chat_message("system", avatar="⚙️"),
        st.expander(title, expanded=not is_complete),
    ):
        # 완료되지 않은 프로세스는 열어두고, 완료되면 접음
        for log in logs:
            st.caption(f"▹ {log}")


@st.fragment
def render_chat_interface():
    """채팅 인터페이스 (Fragment 격리)"""
    chat_container = st.container(height=700, border=False)
    messages = SessionManager.get_messages() or []

    is_generating = bool(SessionManager.get("is_generating_answer", False))

    with chat_container:
        if not messages:
            st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)

        # 시스템 로그 그룹화 렌더링
        current_log_group = []

        for i, msg in enumerate(messages):
            role = cast(str, msg.get("role", "user"))
            content = cast(str, msg.get("content", ""))
            msg_type = cast(str, msg.get("msg_type", "general"))

            if role == "system" or msg_type == "log":
                # [수정] 내부 신호용 로그는 화면에 표시하지 않음
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
                    msg_type=msg_type,
                    msg_index=i,
                )

        if current_log_group:
            _render_system_logs(current_log_group)

    user_query = st.chat_input(MSG_CHAT_INPUT_PLACEHOLDER, disabled=is_generating)

    if user_query and not is_generating:
        SessionManager.add_message("user", user_query)
        st.rerun()

    if not is_generating and messages and messages[-1].get("role") == "user":
        # [수정] UI 스레드의 세션 ID를 명시적으로 추출하여 비동기 스레드에 전달
        from core.rag_core import RAGSystem

        current_sid = SessionManager.get_session_id()
        rag_sys = RAGSystem(session_id=current_sid)

        if SessionManager.is_ready_for_chat(session_id=current_sid):
            SessionManager.set("is_generating_answer", True)
            with chat_container:
                # [수정] 스트리밍 결과를 받아서 세션에 저장
                result = sync_run(
                    _stream_chat_response(
                        rag_sys, messages[-1]["content"], chat_container
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
            st.rerun()
