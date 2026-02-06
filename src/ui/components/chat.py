"""
채팅 인터페이스 및 스트리밍 응답 관련 컴포넌트.
"""

import logging
import re
import time
from contextlib import aclosing
from typing import Any

import streamlit as st

from api.streaming_handler import get_adaptive_controller, get_streaming_handler
from common.config import (
    MSG_CHAT_GUIDE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
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

                    if chunk.metadata and "documents" in chunk.metadata:
                        state["retrieved_docs"] = chunk.metadata["documents"]
                        update_status(
                            f"관련 지식 {len(state['retrieved_docs'])}개 확보"
                        )

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
                            cursor = "▌" if not chunk.is_final else ""
                            answer_area.markdown(display_text + cursor)
                            last_render_time = time.time()

            # [최적화] 스트리밍 종료 후 즉시 결과 반환 (중복 렌더링 제거로 지연 최소화)
            cleaned_final = _clean_response_redundancy(state["full_response"])
            processed_final = apply_tooltips_to_response(
                cleaned_final, state["retrieved_docs"]
            )

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
        SessionManager.add_status_log(friendly_msg)
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
    status_logs: list[str] | None = None,
    is_latest: bool = True,  # [추가] 최신 메시지 여부
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

    # [핵심 최적화] 이미 컨테이너가 있는 경우(스트리밍 최종 단계) 중복 생성을 방지
    msg_container: Any
    if wrap_in_container:
        msg_container = st.chat_message(role, avatar=avatar_icon)
    else:
        # 가짜 컨테이너
        from contextlib import nullcontext

        msg_container = nullcontext()

    with msg_container:
        # 0. 파이프라인 상태 로그 (Status Logs)
        # [최적화] 최신 메시지가 아니면 무거운 st.status 위젯 생성을 생략하여 성능 향상
        if role == "assistant" and status_logs and is_latest:
            with st.status("✅ 분석 완료", state="complete", expanded=False):
                for log in status_logs:
                    if log not in ["시스템 대기 중", "새 문서 분석 시작"]:
                        st.write(f"└─ {log}")

        if thought and thought.strip():
            with st.expander("🧠 사고 완료", expanded=False):
                st.markdown(
                    f'<div class="thought-container">{thought}</div>',
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
                # [리팩토링] 지표를 3개의 컬럼으로 나누어 시각적으로 명확하게 배치
                m_col1, m_col2, m_col3 = st.columns([1, 1, 1.5])

                with m_col1:
                    tokens = metrics.get("token_count", 0)
                    st.markdown(
                        f"📏 **{tokens}** <small>tokens</small>", unsafe_allow_html=True
                    )

                with m_col2:
                    tps = metrics.get("tps", 0)
                    st.markdown(
                        f"🚀 **{tps:.1f}** <small>t/s</small>", unsafe_allow_html=True
                    )

                with m_col3:
                    total = metrics.get("total_time", 0)
                    ttft = metrics.get("ttft", 0)
                    st.markdown(
                        f"⏱️ **{total:.1f}s** <small>(TTFT: {ttft:.2f}s)</small>",
                        unsafe_allow_html=True,
                    )


@st.fragment
def render_chat_interface():
    """채팅 인터페이스 최상위 렌더링 함수 (Fragment 격리)"""
    _chat_fragment()


def _chat_fragment():
    # [최적화] 고정된 Viewport Height 기반의 높이 설정 (JS 의존성 감소)
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

        # [최적화] 대화 이력 렌더링 고속화
        msg_count = len(messages)
        for i, msg in enumerate(messages):
            is_latest = i == msg_count - 1

            if msg.get("role") == "system" or msg.get("msg_type") == "log":
                system_buffer.append(msg["content"])
            else:
                flush_system_buffer()
                render_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    thought=msg.get("thought"),
                    metrics=msg.get("metrics"),
                    processed_content=msg.get("processed_content"),
                    msg_type=msg.get("msg_type", "general"),
                    status_logs=msg.get("status_logs"),
                    is_latest=is_latest,  # [추가] 최신 메시지 여부 전달
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

    if user_query:
        SessionManager.add_message("user", user_query)
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
            processed_final = result.get("processed_content", "")

            if final_answer and not final_answer.startswith("❌"):
                SessionManager.add_message(
                    role="assistant",
                    content=final_answer,
                    processed_content=processed_final,
                    thought=final_thought,
                    metrics=final_metrics,
                    msg_type="answer",
                    status_logs=SessionManager.get("status_logs"),
                    source_file=SessionManager.get("last_uploaded_file_name"),
                    documents=final_docs,
                )
                # [최적화] 프래그먼트 범위 내에서만 리런하여 성능 향상
                st.rerun(scope="fragment")
        else:
            st.error(MSG_CHAT_NO_QA_SYSTEM)
