"""참조 (페이지/doc 점프) 렌더 컴포넌트.

``chat.py`` (P3 분리)에서 이동한 참조 전용 렌더 함수들을 모아둔다:
- ``_handle_page_jump`` / ``_handle_doc_jump`` — 참조 페이지/doc 이동 콜백
- ``_render_citation_anchors`` — citations[] → 클릭 가능한 출처 앵커
- ``_extract_reference_pages`` — 문서 메타데이터에서 참조 페이지 추출
- ``_render_references_content`` — 익스팬더 내 페이지/doc 점프 본문
- ``render_generation_expander`` — 답변 세부정보(메트릭/단계/참조) 통합 익스팬더

본 모듈은 ``chat.py``를 import하지 않는다 (순환 의존 방지). 이동한 함수가
사용하던 모듈 상수/로거가 없으므로 필요한 import만 새로 선언한다.
"""

import contextlib
import html
import time
from collections.abc import Callable
from typing import Any

import streamlit as st

from common.config import DEFAULT_OLLAMA_MODEL
from common.utils import doc_stable_id
from core.session import SessionManager
from ui.components.common import get_doc_metadata, navigate_to_page, status_line
from ui.widget_keys import jump_key

__all__ = [
    "_extract_reference_pages",
    "_handle_doc_jump",
    "_handle_page_jump",
    "_render_citation_anchors",
    "_render_references_content",
    "render_generation_expander",
]


def _handle_page_jump(p: int) -> None:
    """참조 페이지 이동 버튼 콜백입니다."""
    SessionManager.set(
        "pdf_target_page",
        {"page": int(p), "source": "manual", "ts": time.time()},
    )
    navigate_to_page(int(p))
    st.toast(f"Moving to page {p}...")
    # 전체 리런이 필요하다 (뷰어 fragment가 run_every=2.0으로 폴링 중이어도
    # popover 점프는 즉시 반영되어야 하므로 st.rerun()으로 전체 재실행).
    st.rerun()


def _handle_doc_jump(doc_id: str) -> None:
    """인용의 안정 doc_id로 문서를 찾아 첫 페이지로 이동합니다."""
    docs = SessionManager.get("documents", []) or []
    target_page = 1
    found = False
    for d in docs:
        if doc_stable_id(d) == doc_id:
            found = True
            page = get_doc_metadata(d).get("page")
            with contextlib.suppress(ValueError, TypeError):
                if page is not None:
                    target_page = int(page)
            break
    if not found:
        # doc_id만으로 페이지를 알 수 없으면 1p 기준으로 이동한다.
        target_page = 1
    SessionManager.set(
        "pdf_target_page",
        {"page": target_page, "source": "citation", "ts": time.time()},
    )
    navigate_to_page(target_page)
    st.toast("Moving to cited document...")
    st.rerun()


def _render_citation_anchors(
    citations: list[dict[str, Any]], documents: list[Any] | None
) -> None:
    """citations[] 배열을 클릭 가능한 출처 앵커로 렌더합니다.

    PRIMARY 소스는 citations[] (안정 doc_id 기반)이며, 인라인 [doc:N] 폴백은
    apply_tooltips_to_response가 담당합니다. doc_id가 documents에서 실제 문서를
    가리키면 점프 버튼을 노출합니다.
    """
    if not citations:
        return
    doc_ids = {doc_stable_id(d) for d in (documents or [])}
    with st.container():
        st.caption("Sources")
        for idx, cit in enumerate(citations):
            sid = str(cit.get("doc_id", ""))
            span = cit.get("text_span") or cit.get("section") or f"Source {idx + 1}"
            label = html.escape(str(span))[:160]
            if sid in doc_ids:
                if st.button(
                    f"{idx + 1}. {label}",
                    key=f"cit_doc_{sid}_{idx}",
                    use_container_width=True,
                ):
                    _handle_doc_jump(sid)
            else:
                st.markdown(
                    f'<span data-doc-id="{html.escape(sid)}">{idx + 1}. {label}</span>',
                    unsafe_allow_html=True,
                )


def _extract_reference_pages(documents: list[Any]) -> list[int]:
    """문서 메타데이터에서 참조 페이지 번호 목록을 추출합니다."""
    pages: set[int] = set()
    for d in documents:
        meta = get_doc_metadata(d)
        with contextlib.suppress(ValueError, TypeError):
            page = meta.get("page")
            if page is not None:
                pages.add(int(page))
            for pg in meta.get("pages") or []:
                pages.add(int(pg))
    return sorted(pages)


def _render_references_content(
    msg_id: str,
    documents: list[Any] | None,
    on_page_jump: Callable[[int], None] | None = None,
    citations: list[dict[str, Any]] | None = None,
    generating: bool = False,
) -> bool:
    """참조 콘텐츠(페이지/doc 점프 버튼)를 렌더링합니다.

    통합 익스팬더 내부에서 직접 호출되므로 popover 래퍼 없이 본문만 그립니다.
    렌더된 참조가 있으면 True, 없으면 False를 반환합니다.

    generating=True(스트리밍 중)에는 매 chunk rerun마다 동일 위젯이 재생성되므로
    key를 소비하는 st.button 대신 정적 markdown으로 페이지/doc을 표시합니다.
    key가 필요한 상호작용 점프 버튼은 완료 후(generating=False, 단일 rerun)에만
    렌더하므로 StreamlitDuplicateElementKey 충돌을 피합니다.
    """
    rendered = False
    if not documents and not citations:
        return rendered

    pages = _extract_reference_pages(documents or [])
    if pages:
        st.caption("By page")
        if generating:
            # 스트리밍 중: key 없는 정적 표시 (중복 등록 방지).
            st.markdown(" · ".join(f"`{p}p`" for p in pages))
        else:
            cols = st.columns(min(len(pages), 5))
            for idx, p in enumerate(pages):
                clicked = cols[idx % len(cols)].button(
                    f"{p}p",
                    key=jump_key(msg_id, p, idx),
                    use_container_width=True,
                )
                if clicked and on_page_jump is not None:
                    on_page_jump(p)
        rendered = True

    # P3: citations[] 기반 doc 점프 (안정 doc_id).
    doc_citations = [c for c in (citations or []) if c.get("doc_id") is not None]
    if doc_citations:
        doc_ids = {doc_stable_id(d) for d in (documents or [])}
        st.caption("By doc")
        for idx, cit in enumerate(doc_citations):
            sid = str(cit.get("doc_id"))
            if sid in doc_ids:
                label = cit.get("section") or cit.get("text_span") or f"doc {sid}"
                if generating:
                    # 스트리밍 중: key 없는 정적 표시.
                    st.markdown(
                        f'{idx + 1}. <span data-doc-id="{html.escape(sid)}">'
                        f"{html.escape(label)}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        f"{idx + 1}. {label}",
                        key=f"pop_doc_{msg_id}_{sid}_{idx}",
                        use_container_width=True,
                    ):
                        _handle_doc_jump(sid)
        rendered = True
    return rendered


def render_generation_expander(
    msg: dict[str, Any],
    *,
    expanded: bool,
    generating: bool,
    status_text: str = "Answer generation",
    process_override: dict[str, Any] | None = None,
) -> None:
    """답변 말풍선 상단(질문↔답변 사이)에 **단일 고정 익스팬더**를 렌더합니다.

    메트릭·생성 단계·상위 점수·사고 과정·참조를 모두 이 익스팬더 안에 수납해
    산개되던 부가 정보(별도 Metrics 익스팬더, References popover, 완료 후
    generation 익스팬더)를 하나로 통합한다. 기본값은 접힘(expanded=False).

    스트리밍 중과 완료 후 동일 위젯을 재사용해, 생성 완료 시 상태 박스가
    증발하던 문제를 해결한다. 본문은 매 렌더 **무조건** 작성하므로 fragment
    폴링(0.5s)으로 st.expander가 재생성되어도 내용이 비지 않는다.

    - generating=True: 기본 접힘 유지, 내부 st.spinner로 진행 표시
    - generating=False: 접은 상태(완료 후 유지), 정적 헤더만
    - cancelled 메시지는 추론 로그를 감춰 전체 추론 완료로 오인되지 않게 함
    - process_override: 완료 메시지처럼 이미 계산된 process dict가 있으면
      재파생(process_steps 의존) 대신 직접 사용한다.
    """
    thought = msg.get("thought", "") or ""
    cancelled = bool(msg.get("cancelled", False))
    show_thought = bool(thought and thought.strip() and not cancelled)
    documents = msg.get("documents") or []
    citations = msg.get("citations") or []
    metrics = msg.get("metrics") or {}
    msg_id = msg.get("msg_id") or ""

    # ui.components 내부 순환 의존을 피하기 위해 lazy import
    from ui.components.streaming import _build_process

    process = process_override or _build_process(msg) or {}
    steps = process.get("steps") or []
    sections = process.get("sections") or []
    top_scores = [
        s
        for s in (process.get("top_scores") or [])
        if isinstance(s, dict) and "section" in s and "score" in s
    ]
    perf = process.get("perf") or {}

    # 메트릭(완료 메시지에 실린 metrics)도 익스팬더 수납 대상.
    retrieved = (
        len(documents) if documents else (process or {}).get("retrieved_count", 0)
    )
    total_time = metrics.get("total_time", 0)
    has_metrics = bool(total_time or retrieved)
    has_block = bool(
        steps or sections or top_scores or perf or show_thought or has_metrics
    )

    # 완료 메시지인데 표시할 내용이 없으면 익스팬더 자체를 렌더하지 않는다.
    # 빈 익스팬더 헤더가 대화 줄 간격(패딩+익스팬더)을 키워 간격 과대를 유발한다.
    # 생성 중(generating=True)에는 항상 익스팬더를 열어 진행 표시/깜빡임을 방지한다.
    if not generating and not (has_block or documents or citations):
        return

    with st.expander("Answer details", expanded=expanded):
        if generating:
            with st.spinner(status_text):
                pass  # spinner는 헤더 아래 진행 표시용(본문은 아래 즉시 작성)

        if not (has_block or documents or citations):
            # 생성 중인데 아직 표시할 내용이 없으면 진행 캡션만 노출.
            st.caption("Preparing...")
            return

        if steps:
            st.markdown(" · ".join(steps))
        if sections:
            st.caption(" · ".join(sections))
        if top_scores:
            st.caption(
                ", ".join(f"{s['section']} {s['score']:.3f}" for s in top_scores)
            )

        # 메트릭: Time / Retrieved / Model (UX-3: 기본 접힘 익스팬더 내 수납).
        parts = []
        if isinstance(total_time, (int, float)):
            parts.append(f"Time: {total_time:.1f}s")
        if retrieved:
            parts.append(f"Retrieved: {retrieved} chunks")
        model = (
            msg.get("model", "")
            or SessionManager.get("last_selected_model", "")
            or DEFAULT_OLLAMA_MODEL
        )
        if model:
            parts.append(f"Model: {model}")
        if parts:
            st.caption(status_line(*parts))

        if show_thought:
            st.markdown("**Thinking process**")
            st.markdown(thought)

        # 참조(페이지/doc 점프) — 기존 References popover 내용을 익스팬더 안으로 통합.
        if documents or citations:
            st.divider()
            st.caption("References")
            _render_references_content(
                msg_id,
                documents,
                on_page_jump=_handle_page_jump,
                citations=citations,
                generating=generating,
            )
