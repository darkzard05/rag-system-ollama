"""
채팅 인터페이스 및 스트리밍 응답 관련 컴포넌트 - Native Streaming Refactored
"""

import asyncio
import contextlib
import logging
import queue
import re
import threading
from typing import Any, TypedDict

import streamlit as st

from api.streaming_handler import get_streaming_handler
from common.config import (
    MSG_CHAT_GUIDE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_THINKING,
)
from common.utils import (
    apply_tooltips_to_response,
    extract_annotations_from_docs,
    format_error_message,
    normalize_latex_delimiters,
)
from core.rag_core import RAGSystem
from core.session import SessionManager

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    full_response: str
    full_thought: str
    retrieved_docs: list[Any]
    performance: dict[str, Any]
    thinking_start_time: float
    thinking_end_time: float


def _sync_stream_generator(query: str, model_name: str, session_id: str):
    """비동기 스트림을 동기 Streamlit 환경에서 소비하기 위한 브릿지 제너레이터"""
    q = queue.Queue()

    def bg_task():
        async def run():
            try:
                SessionManager.set_session_id(session_id)
                rag_sys = RAGSystem(session_id=session_id)
                event_generator = await rag_sys.astream(query, model_name=model_name)
                handler = get_streaming_handler()
                event_stream = handler.stream_graph_events(event_generator)

                async for chunk in event_stream:
                    q.put(("chunk", chunk))
            except Exception as e:
                q.put(("error", e))
            finally:
                q.put(("done", None))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
        loop.close()

    t = threading.Thread(target=bg_task, daemon=True)
    t.start()

    while True:
        msg_type, data = q.get()
        if msg_type == "done":
            break
        elif msg_type == "error":
            raise data
        else:
            yield data


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
    is_latest: bool = False,
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
            st.markdown(
                f"""
                <details class="thought-expander">
                    <summary>{MSG_THINKING[:-3]} 완료</summary>
                    <div class="thought-container">{thought}</div>
                </details>
                """,
                unsafe_allow_html=True,
            )

        if processed_content:
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            display_text = normalize_latex_delimiters(content)
            if role == "assistant" and documents:
                display_text = apply_tooltips_to_response(display_text, documents)
            st.markdown(display_text, unsafe_allow_html=(role == "assistant"))

        if role == "assistant" and (metrics or documents):
            with st.popover("ℹ️ 상세 정보 및 참조", use_container_width=False):
                if metrics:
                    total = metrics.get("total_time", 0)
                    tps = metrics.get("tps", metrics.get("tokens_per_second", 0))
                    in_tok = metrics.get("input_token_count", 0)
                    out_tok = metrics.get(
                        "token_count", metrics.get("output_token_count", 0)
                    )
                    st.caption(
                        f"**성능 지표:** ⏱️ {total:.1f}s ({tps:.1f} tok/s) | 📥 In: {in_tok} | 📤 Out: {out_tok}"
                    )

                if documents:
                    extracted_pages = set()
                    for d in documents:
                        meta = (
                            getattr(d, "metadata", {})
                            if hasattr(d, "metadata")
                            else d.get("metadata", {})
                        )
                        p = meta.get("page", 1)
                        with contextlib.suppress(ValueError, TypeError):
                            extracted_pages.add(int(p))

                    pages = sorted(extracted_pages)
                    if pages:
                        st.divider()
                        st.caption("**📑 참조 페이지로 이동:**")
                        cols = st.columns(min(len(pages), 5))
                        for idx, p in enumerate(pages):
                            if cols[idx % len(cols)].button(
                                f"{p}p",
                                key=f"jump_{msg_id}_{p}_{idx}",
                                use_container_width=True,
                            ):
                                SessionManager.set("pdf_target_page", p)
                                SessionManager.set("current_page", p)
                                st.rerun()


@st.fragment(run_every=1.0)
def _render_build_status_in_chat(sid: str):
    """문서 분석 진행 상황을 채팅창 상단에 표시합니다."""
    is_building = bool(SessionManager.get("is_building_rag", False, sid))

    if is_building:
        status_msg = SessionManager.get("rebuild_status", "문서 분석 중...", sid)
        with st.status(f"⏳ {status_msg}", expanded=True):
            logs = SessionManager.get("status_logs", [], sid)
            if logs:
                for log in logs[-3:]:
                    st.caption(f"▹ {log}")
            else:
                st.write("파이프라인 구축을 시작합니다...")

        rebuild_done = bool(SessionManager.get("rebuild_done", False, sid))
        if rebuild_done:
            SessionManager.set("is_building_rag", False, sid)
            st.rerun(scope="app")


def render_chat_interface():
    messages = SessionManager.get_messages() or []
    current_sid = SessionManager.get_session_id()
    is_generating = bool(SessionManager.get("is_generating_answer", False, current_sid))
    model_name = SessionManager.get("last_selected_model", session_id=current_sid)

    _render_build_status_in_chat(current_sid)

    # [수정] 클래스 부여를 위해 외곽 컨테이너 추가 및 하드코딩된 height 제거
    with st.container():
        st.markdown('<div class="chat-scroll-container">', unsafe_allow_html=True)

        # [주의] st.container(height=...) 대신 CSS 클래스 제어를 위해 일반 container 사용
        with st.container(border=False):
            if not messages:
                st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)

            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if (
                    role == "system"
                    or msg.get("msg_type") == "log"
                    or content == "READY_FOR_QUERY"
                ):
                    continue

                is_latest = (i == len(messages) - 1) and not is_generating
                render_message(
                    role=role,
                    content=content,
                    thought=msg.get("thought"),
                    documents=msg.get("documents"),
                    metrics=msg.get("metrics"),
                    processed_content=msg.get("processed_content"),
                    msg_index=i,
                    msg_id=msg.get("msg_id"),
                    is_latest=is_latest,
                )

            if is_generating and messages and messages[-1].get("role") == "user":
                query = messages[-1]["content"]

                with st.chat_message("assistant", avatar="🤖"):
                    stream_placeholder = st.empty()

                    content_acc = ""
                    thought_acc = ""
                    docs_acc = []
                    perf_acc = {}
                    current_status = ""

                    try:
                        for chunk in _sync_stream_generator(
                            query, model_name, current_sid
                        ):
                            if chunk.status:
                                # [개선] 상태 메시지에 streaming-pulse 클래스 적용하여 레이아웃 점프 방지
                                current_status = f"<div class='streaming-pulse'>⏳ {chunk.status}</div>\n\n"
                            if chunk.metadata and "documents" in chunk.metadata:
                                docs_acc = chunk.metadata["documents"]
                            if chunk.performance:
                                perf_acc = chunk.performance
                            if chunk.thought:
                                thought_acc += chunk.thought
                            if chunk.content:
                                content_acc += chunk.content

                            display_text = normalize_latex_delimiters(content_acc)
                            thought_html = (
                                f"""
                            <details class="thought-expander" open>
                                <summary>🤔 생각 과정</summary>
                                <div class="thought-container">{thought_acc}</div>
                            </details>
                            """
                                if thought_acc
                                else ""
                            )

                            stream_placeholder.markdown(
                                f"{current_status}{thought_html}{display_text} ▌",
                                unsafe_allow_html=True,
                            )

                        final_content = _clean_response_redundancy(content_acc)
                        processed_content = apply_tooltips_to_response(
                            final_content, docs_acc
                        )

                        thought_html_final = (
                            f"""
                        <details class="thought-expander">
                            <summary>{MSG_THINKING[:-3]} 완료</summary>
                            <div class="thought-container">{thought_acc}</div>
                        </details>
                        """
                            if thought_acc
                            else ""
                        )

                        stream_placeholder.markdown(
                            f"{thought_html_final}{processed_content}",
                            unsafe_allow_html=True,
                        )

                        SessionManager.add_message(
                            role="assistant",
                            content=final_content,
                            thought=thought_acc,
                            documents=docs_acc,
                            metrics=perf_acc,
                            processed_content=processed_content,
                            session_id=current_sid,
                        )

                        if docs_acc:
                            annotations = extract_annotations_from_docs(docs_acc)
                            SessionManager.set(
                                "pdf_annotations", annotations, current_sid
                            )
                            try:
                                target_p = getattr(docs_acc[0], "metadata", {}).get(
                                    "page"
                                )
                                if target_p:
                                    SessionManager.set(
                                        "pdf_target_page", int(target_p), current_sid
                                    )
                                    SessionManager.set(
                                        "current_page", int(target_p), current_sid
                                    )
                            except Exception:
                                pass

                    except Exception as e:
                        logger.error(f"Streaming error: {e}", exc_info=True)
                        error_msg = format_error_message(e)
                        stream_placeholder.error(error_msg)
                        SessionManager.add_message(
                            "assistant", error_msg, session_id=current_sid
                        )
                    finally:
                        SessionManager.set("is_generating_answer", False, current_sid)
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    input_placeholder = (
        MSG_CHAT_INPUT_PLACEHOLDER if not messages else "추가 질문을 입력하세요..."
    )
    user_query = st.chat_input(
        input_placeholder, disabled=is_generating, key="main_chat_input"
    )

    if user_query and not is_generating:
        SessionManager.add_message("user", user_query, session_id=current_sid)
        SessionManager.set("is_generating_answer", True, current_sid)
        st.rerun()
