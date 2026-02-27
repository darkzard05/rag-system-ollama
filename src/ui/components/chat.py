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

    # [Lazy Import]
    from core.rag_core import RAGSystem

    rag_sys = RAGSystem(session_id=SessionManager.get_session_id())

    SessionManager.set("is_generating_answer", True)

    handler = get_streaming_handler()
    controller = get_adaptive_controller()

    last_render_time = 0.0
    render_interval = 0.05

    try:
        with chat_container, st.chat_message("assistant", avatar="🤖"):
            # [수정] 문서 분석 타임라인과 동일한 스타일의 실시간 상태 박스
            status_placeholder = st.empty()
            live_logs = []

            def update_status(msg: str, is_complete=False):
                if msg not in live_logs:
                    live_logs.append(msg)

                status_icon = "✅" if is_complete else "🚀"
                # 상태 메시지에서 '파이프라인 가동 중:' 접두사를 제거하여 더 깔끔하게 표시
                current_label = msg.split(": ", 1)[-1] if ": " in msg else msg
                expander_title = f"{status_icon} {current_label}"

                lines = "".join(
                    [
                        f"<div style='font-size: 0.85rem; color: var(--text-color); margin-bottom: 8px; display: flex; align-items: flex-start; line-height: 1.5; opacity: 0.9;'>"
                        f"<span style='color: #1e88e5; margin-right: 10px; font-weight: bold;'>▹</span>"
                        f"<span>{item}</span></div>"
                        for item in live_logs
                    ]
                )

                timeline_html = (
                    f"<div style='margin-bottom: 15px;'>"
                    f"<details {'open' if not is_complete else ''} class='timeline-container' style='border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 10px;'>"
                    f"<summary class='timeline-summary' style='font-weight: 600; color: var(--text-color); cursor: pointer; list-style: none; display: flex; align-items: center; padding: 5px 0;'>"
                    f"{expander_title}</summary>"
                    f"<div style='margin-top: 12px; padding: 15px; background-color: rgba(128,128,128,0.05); border-radius: 8px; border-left: 3px solid #1e88e5;'>"
                    f"<div style='font-size: 0.75rem; color: var(--text-color); opacity: 0.6; margin-bottom: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>"
                    f"⏱️ Live Pipeline Execution"
                    f"</div>{lines}</div></details></div>"
                )
                status_placeholder.markdown(timeline_html, unsafe_allow_html=True)

            thought_area = st.container()
            answer_area = st.empty()

            # [핵심] 루프 진입 전 초기 상태 즉시 표시
            update_status("🚀 질문 분석 및 파이프라인 가동 중...")

            # RAGSystem 인터페이스를 통해 이벤트 스트림 획득
            event_generator = await rag_sys.astream_events(user_query, llm=current_llm)

            async with aclosing(  # type: ignore[type-var]
                handler.stream_graph_events(
                    event_generator, adaptive_controller=controller
                )
            ) as stream:
                async for chunk in stream:
                    if chunk.status:
                        # [개선] 스트리밍 핸들러가 제공하는 상세 상태를 그대로 활용
                        update_status(chunk.status)
                        SessionManager.add_status_log(chunk.status)

                    if chunk.metadata and "documents" in chunk.metadata:
                        state["retrieved_docs"] = chunk.metadata["documents"]
                        doc_msg = f"📚 관련 지식 {len(state['retrieved_docs'])}개 확보 및 검증 완료"
                        SessionManager.add_status_log(doc_msg)
                        update_status(doc_msg)

                    if chunk.performance:
                        state["performance"] = chunk.performance

                    if chunk.thought:
                        if not state["full_thought"]:
                            state["thinking_start_time"] = time.time()
                            thought_msg = "🧠 AI가 최적의 답변 논리를 설계 중..."
                            SessionManager.add_status_log(thought_msg)
                            update_status(thought_msg)
                            with thought_area:
                                st.caption("AI of Thought:")
                                thought_display = st.empty()
                        state["full_thought"] += chunk.thought
                        if time.time() - last_render_time > render_interval:
                            thought_display.markdown(f"*{state['full_thought']}*")
                            last_render_time = time.time()

                    if chunk.content:
                        if not state["full_response"]:
                            gen_msg = "✍️ 지식 기반으로 최적의 답변 작성 시작"
                            SessionManager.add_status_log(gen_msg)
                            update_status(gen_msg, is_complete=True)
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
            SessionManager.add_status_log("✨ 답변 생성 완료")
            cleaned_final = _clean_response_redundancy(state["full_response"])
            processed_final = apply_tooltips_to_response(
                cleaned_final, state["retrieved_docs"]
            )

            # --- [추가] 자동 점프 트리거 설정 ---
            if state["retrieved_docs"]:
                try:
                    # 가장 관련성 높은 첫 번째 문서의 페이지 추출
                    first_doc = state["retrieved_docs"][0]
                    # Document 객체 또는 dict 형태 모두 대응
                    if hasattr(first_doc, "metadata"):
                        metadata = first_doc.metadata
                    else:
                        metadata = first_doc.get("metadata", {})

                    target_p = metadata.get("page")
                    if target_p is not None:
                        # 0-indexed일 경우를 대비해 1-indexed로 보정 (라이브러리에 따라 다름)
                        # 보통 LangChain/PyMuPDF는 0부터 시작하는 경우가 많음
                        st.session_state.pdf_target_page = int(target_p) + 1
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
    is_latest: bool = True,
    msg_index: int = 0,
    **kwargs,
):
    """메시지를 렌더링하는 통합 엔진. msg_type에 따라 레이아웃 자동 결정."""

    if role == "system" or msg_type == "log":
        with st.chat_message("system", avatar="⚙️"):
            if msg_type == "log":
                st.markdown(
                    f"<div style='font-size: 0.85rem; color: var(--text-color); opacity: 0.7;'>└─ {content}</div>",
                    unsafe_allow_html=True,
                )
            elif "완료" in content or "성공" in content:
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
        # 0. [중복 제거] 기존 assistant 내 status_logs(st.status) 위젯 제거
        # 모든 로그는 상단 시스템 타임라인에서 통합 관리함

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
                m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1.5])

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
                    doc_count = metrics.get("doc_count", 0)
                    if documents and len(documents) > 0:
                        # 페이지 번호 추출 및 중복 제거
                        pages = []
                        for d in documents:
                            # Document 객체 또는 dict 대응
                            if hasattr(d, "metadata"):
                                m = d.metadata
                            else:
                                m = d.get("metadata", {})

                            p = m.get("page")
                            if p is not None:
                                # [수정] 이미 1-indexed이므로 그대로 사용
                                pages.append(int(p))

                        unique_pages = sorted(set(pages))

                        if unique_pages:
                            # 팝오버를 사용하여 깔끔하게 표시
                            with st.popover(
                                f"📄 {doc_count} Docs", use_container_width=True
                            ):
                                st.caption("근거 페이지로 이동:")
                                cols = st.columns(min(len(unique_pages), 3))
                                for idx, p in enumerate(unique_pages):
                                    # [수정] 고정된 키 사용 (time.time() 제거) 및 확실한 이벤트 캡처
                                    button_key = f"jump_btn_{msg_index}_{p}_{idx}"
                                    if cols[idx % 3].button(f"{p}p", key=button_key):
                                        logger.info(
                                            f"[DEBUG] 페이지 점프 실행: {p}p (Key: {button_key})"
                                        )
                                        SessionManager.set("pdf_target_page", p)
                                        st.rerun()
                        else:
                            st.markdown(
                                f"📄 **{doc_count}** <small>Docs</small>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            f"📄 **{doc_count}** <small>Docs</small>",
                            unsafe_allow_html=True,
                        )

                with m_col4:
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
    # [수정] 자동 스크롤과 하단 고정을 위해 고정 높이 컨테이너 사용
    # height를 지정하면 내용이 늘어날 때 자동으로 하단을 추적합니다.
    chat_container = st.container(height=700, border=False)

    messages = SessionManager.get_messages() or []
    pdf_path = SessionManager.get("pdf_file_path")
    pdf_processed = SessionManager.get("pdf_processed", False)
    pdf_error = SessionManager.get("pdf_processing_error")

    is_generating = bool(SessionManager.get("is_generating_answer", False))
    is_processing_pdf = bool(pdf_path and not pdf_processed and not pdf_error)

    with chat_container:
        if not messages:
            st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)

        system_buffer: list[str] = []

        def flush_system_buffer():
            if not system_buffer:
                return

            # [수정] 주요 과정 선별 기준 최적화
            MAJOR_STEPS = {
                "📑": "문서 구조 분석 및 마크다운 변환",
                "✂️": "문서 분할 및 지식 청킹",
                "🧠": "지식 벡터화 및 인덱싱",
                "🔍": "질문 의도 분석 및 하이브리드 검색",
                "📚": "관련 지식 확보 및 문서 검증",
                "⚖️": "문서 순위 재조정 및 적합도 검증",
                "🎯": "핵심 답변 근거 선정 및 컨텍스트 정제",
                "🧩": "답변용 지식 컨텍스트 병합",
                "✍️": "지식 기반으로 최적의 답변 작성 시작",
            }

            is_doc_analysis = False
            is_complete = False
            has_error = False
            log_items: list[str] = []

            for m in system_buffer:
                # 1. 완료 및 오류 상태 확인
                if m == "READY_FOR_QUERY" or "완료" in m or "성공" in m:
                    is_complete = True
                if any(x in m for x in ["❌", "오류", "실패"]):
                    has_error = True

                # 2. 주요 단계 매칭 (아이콘 우선 매칭)
                matched = False
                for icon, label in MAJOR_STEPS.items():
                    if icon in m:
                        # 같은 아이콘이 이미 있으면 내용을 보고 결정 (아이콘은 같은데 내용이 다르면 추가)
                        if not any(icon in li for li in log_items):
                            log_items.append(f"{icon} {label}")
                            if icon in ["📑", "✂️", "🧠"]:
                                is_doc_analysis = True
                            if icon in ["🔍", "📚", "⚖️", "🎯", "🧩", "✍️"]:
                                pass
                        matched = True
                        break

                # 3. 아이콘 매칭 안 된 경우 키워드 매칭
                if not matched:
                    if any(x in m for x in ["분석", "마크다운", "구조"]) and not any(
                        "📑" in li for li in log_items
                    ):
                        log_items.append(f"📑 {MAJOR_STEPS['📑']}")
                        is_doc_analysis = True
                    elif ("벡터화" in m or "인덱싱" in m) and not any(
                        "🧠" in li for li in log_items
                    ):
                        log_items.append(f"🧠 {MAJOR_STEPS['🧠']}")
                        is_doc_analysis = True

            if log_items:
                # [개선] 마지막 단계를 제목으로 사용 (동적 제목 시스템)
                current_step_label = log_items[-1]
                is_expanded = not is_complete and not has_error

                # 상태에 따른 아이콘 및 접두사 결정
                if has_error:
                    status_prefix = "❌ 오류: "
                elif is_complete:
                    status_prefix = "✅ 완료: "
                else:
                    status_prefix = "⚙️ 처리 중: "

                # 최종 제목 구성 (아이콘 제외 텍스트만 추출하여 조합)
                clean_label = current_step_label.split(" ", 1)[-1]
                expander_title = f"{status_prefix}{clean_label}"

                with st.chat_message("system", avatar="⚙️"):
                    # [수정] 다크모드 시인성을 위해 하드코딩된 색상 제거 및 테마 변수 활용
                    # [수정] 다크모드 시인성을 위해 하드코딩된 색상 제거 및 테마 변수 활용
                    lines = "".join(
                        [
                            f"<div style='font-size: 0.85rem; color: var(--text-color); margin-bottom: 8px; display: flex; align-items: flex-start; line-height: 1.5; opacity: 0.9;'>"
                            f"<span style='color: #1e88e5; margin-right: 10px; font-weight: bold;'>▹</span>"
                            f"<span>{item}</span></div>"
                            for item in log_items
                        ]
                    )

                    timeline_html = (
                        f"<details {'open' if is_expanded else ''} class='timeline-container' style='border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 10px; margin-bottom: 10px;'>"
                        f"<summary class='timeline-summary' style='font-weight: 600; color: var(--text-color); cursor: pointer; list-style: none; display: flex; align-items: center; padding: 5px 0;'>"
                        f"<span style='color: #1e88e5; margin-right: 10px;'>{'✅' if is_complete else '⚙️'}</span> {expander_title.split(': ', 1)[-1] if ': ' in expander_title else expander_title}"
                        f"</summary>"
                        f"<div style='margin-top: 12px; padding: 15px; background-color: rgba(128,128,128,0.05); border-radius: 8px; border-left: 3px solid #1e88e5;'>"
                        f"<div style='font-size: 0.75rem; color: var(--text-color); opacity: 0.6; margin-bottom: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>"
                        f"⏱️ {'Document Analysis' if is_doc_analysis else 'Answer Generation'} Sequence"
                        f"</div>{lines}</div></details>"
                    )
                    st.markdown(timeline_html, unsafe_allow_html=True)

            system_buffer.clear()

        # [최적화] 대화 이력 렌더링 고속화
        msg_count = len(messages)
        for i, msg in enumerate(messages):
            is_latest = i == msg_count - 1

            if msg.get("role") == "system" or msg.get("msg_type") == "log":
                system_buffer.append(str(msg.get("content", "")))
            else:
                flush_system_buffer()

                # 타입 안전성을 위해 명시적 추출
                msg_metrics = msg.get("metrics")
                msg_logs = msg.get("status_logs")

                render_message(
                    role=str(msg.get("role", "user")),
                    content=str(msg.get("content", "")),
                    thought=msg.get("thought"),
                    documents=msg.get("documents"),
                    metrics=msg_metrics if isinstance(msg_metrics, dict) else None,
                    processed_content=msg.get("processed_content"),
                    msg_type=str(msg.get("msg_type", "general")),
                    status_logs=msg_logs if isinstance(msg_logs, list) else None,
                    is_latest=is_latest,
                    msg_index=i,  # 인덱스 전달
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
            # [안전 장치] 답변 생성 시작 시 플래그 설정 및 오류 발생 시 복구 보장
            SessionManager.set("is_generating_answer", True)
            try:
                result = sync_run(
                    _stream_chat_response(rag_engine, user_query, chat_container)
                )

                final_answer = result.get("response", "")
                final_thought = result.get("thought", "")
                final_docs = result.get("documents", [])
                final_metrics = result.get("performance")
                processed_final = result.get("processed_content", "")

                if final_answer and not final_answer.startswith("❌"):
                    # [추가] 답변 생성에 사용된 문서들을 기반으로 PDF 하이라이트 생성
                    from common.utils import extract_annotations_from_docs

                    annotations = extract_annotations_from_docs(final_docs)
                    SessionManager.set("pdf_annotations", annotations)

                    # 상세 로깅 (디버깅 용도)
                    pages = sorted({a["page"] + 1 for a in annotations})
                    logger.info(
                        f"[UI] PDF 하이라이트 적용 완료: {len(annotations)}개 영역 (Pages: {pages})"
                    )

                    SessionManager.add_message(
                        role="assistant",
                        content=final_answer,
                        processed_content=processed_final,
                        thought=final_thought,
                        metrics=final_metrics,
                        msg_type="answer",
                        documents=final_docs,
                        status_logs=SessionManager.get("status_logs"),
                        source_file=SessionManager.get("last_uploaded_file_name"),
                    )
            except Exception as e:
                logger.error(f"채팅 처리 중 예외 발생: {e}", exc_info=True)
                st.error(f"시스템 오류가 발생했습니다: {e}")
            finally:
                # [핵심] 어떤 경우에도 생성 중 플래그 해제
                SessionManager.set("is_generating_answer", False)
                # [최적화] 프래그먼트 범위 내에서만 리런하여 성능 향상
                st.rerun(scope="fragment")
        else:
            st.error(MSG_CHAT_NO_QA_SYSTEM)
