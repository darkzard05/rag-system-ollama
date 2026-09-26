"""참조 (페이지/doc 점프) 렌더 컴포넌트.

``chat.py`` (P3 분리)에서 이동한 참조 전용 렌더 함수들을 모아둔다:
- ``_handle_page_jump`` / ``_handle_doc_jump`` — 참조 페이지/doc 이동 콜백
- ``_extract_reference_pages`` — 문서 메타데이터에서 참조 페이지 추출
- ``_render_references_content`` — 익스팬더 내 페이지/doc 점프 본문
- ``render_generation_expander`` — 답변 세부정보(메트릭/단계/참조) 통합 익스팬더

본 모듈은 ``chat.py``를 import하지 않는다 (순환 의존 방지). 이동한 함수가
사용하던 모듈 상수/로거가 없으므로 필요한 import만 새로 선언한다.
"""

import contextlib
import html
import re
import time
from typing import Any

import streamlit as st

from common.utils import doc_stable_id, fast_hash
from core.session import SessionManager
from ui.components.common import get_doc_metadata, navigate_to_page
from ui.strings import t

__all__ = [
    "_extract_reference_pages",
    "_handle_doc_jump",
    "_handle_page_jump",
    "_render_references_content",
    "normalize_excerpt",
    "render_generation_expander",
    "render_inline_citation_badges",
]


def normalize_excerpt(text: object, max_chars: int = 200) -> str:
    """호버용 발췌 정규화: strip + 공백 축소 + ~200자 절단.

    Streamlit ``help=`` 툴팁은 ``\\n\\n``에서 깨지므로(#13339) 모든 개행을
    제거해 단일 행으로 만든다. 배지/점프 버튼 렌더 경로가 공유한다.
    """
    if not isinstance(text, str):
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) > max_chars:
        return collapsed[:max_chars].rstrip() + "..."
    return collapsed


def _doc_raw_text_map(documents: list[Any] | None) -> dict[str, str]:
    """doc 안정 ID -> 원문(page_content) 매핑 (발췌 표시용, 정규화 없음)."""
    raw_map: dict[str, str] = {}
    for d in documents or []:
        sid = doc_stable_id(d)
        if sid in raw_map:
            continue
        content = getattr(d, "page_content", None)
        if content is None and isinstance(d, dict):
            content = d.get("page_content")
        raw_map[sid] = content if isinstance(content, str) else ""
    return raw_map


def render_inline_citation_badges(
    citations: list[dict[str, Any]] | None,
    documents: list[Any] | None = None,
) -> bool:
    """답변 말미 짧은 회색 배지 ``[1..N]`` 렌더 (발췌는 ``help=`` 호버).

    클릭 불가한 파란 전문 블록을 대체한다. 발췌원은 ``text_span`` 우선,
    없으면 동일 ``doc_id`` 문서 원문, 마지막으로 섹션명이다.
    """
    doc_citations = [c for c in (citations or []) if c.get("doc_id") is not None]
    if not doc_citations:
        return False
    raw_map = _doc_raw_text_map(documents)
    for idx, cit in enumerate(doc_citations):
        sid = str(cit.get("doc_id"))
        span = cit.get("text_span")
        raw = span if isinstance(span, str) and span else raw_map.get(sid, "")
        if not raw:
            section = cit.get("section")
            raw = section if isinstance(section, str) else ""
        excerpt = normalize_excerpt(raw) or f"Source {idx + 1}"
        st.badge(f"[{idx + 1}]", color="grey", help=excerpt)
    return True


def _handle_page_jump(page: int | str) -> None:
    """인용 출처 클릭 시 해당 PDF 페이지로 이동하는 콜백."""
    with contextlib.suppress(ValueError, TypeError):
        target_page = max(1, int(page))
        SessionManager.set(
            "pdf_target_page",
            {"page": target_page, "source": "manual", "ts": time.time()},
        )
        navigate_to_page(target_page)
        # 좌측 PDF 뷰어와 페이지 컨트롤이 즉시 갱신되므로 4초간 화면을 가리는 불필요한 토스트 제거


# 하위 호환성 유지를 위한 alias (기존 참조 보호)
_handle_doc_jump = _handle_page_jump


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


def _normalize_text_for_dedup(text: object) -> str:
    """Branch-D 제거용 본문 정규화: 연속 공백 축소 + 소문자화."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _render_references_content(
    msg_id: str,
    documents: list[Any] | None,
    citations: list[dict[str, Any]] | None = None,
    generating: bool = False,
    show_caption: bool = True,
) -> bool:
    rendered = False
    if not documents and not citations:
        return rendered

    # 1. 인용 데이터 수집 (기존 로직 유지)
    doc_citations = [c for c in (citations or []) if c.get("doc_id") is not None]
    if not doc_citations and documents:
        seen_keys = set()
        fallback_citations = []
        for d in documents:
            meta = get_doc_metadata(d)
            sec = meta.get("current_section", "문서 본문")
            page = meta.get("page", 1)
            key = (sec, page)
            if key not in seen_keys:
                seen_keys.add(key)
                fallback_citations.append(
                    {"doc_id": doc_stable_id(d), "section": sec, "page": page}
                )
        doc_citations = fallback_citations

    if not doc_citations:
        return False

    # 2. 문서 ID별 페이지 매핑 테이블 구축
    doc_page_map = {}
    for d in documents or []:
        p = get_doc_metadata(d).get("page")
        if p is not None:
            with contextlib.suppress(ValueError, TypeError):
                doc_page_map[doc_stable_id(d)] = int(p)

    # 2b. Branch-D: 문서 ID별 정규화 본문 매핑 (동일 텍스트 판별용)
    doc_text_map: dict[str, str] = {}
    for d in documents or []:
        sid_d = doc_stable_id(d)
        if sid_d in doc_text_map:
            continue
        content = getattr(d, "page_content", None)
        if content is None and isinstance(d, dict):
            content = d.get("page_content")
        doc_text_map[sid_d] = _normalize_text_for_dedup(content)

    # 2c. 호버 발췌용 원문 매핑 (정규화 없음 — help= 표시용)
    doc_raw_map = _doc_raw_text_map(documents)

    # 3. [개선] 페이지 번호(target_p) 및 대표 섹션 기준 그룹화 (중복 버튼 방지)
    # Branch-D: 동일 정규화 텍스트가 여러 페이지에 나타나면 최소 페이지만
    # 유지하고, 병합 시 가장 구체적인(긴) 섹션명을 보존한다.
    page_to_refs: dict[int, dict[str, Any]] = {}
    page_to_excerpt: dict[int, str] = {}
    seen_text_hashes: dict[str, int] = {}
    targets: list[tuple[int, dict[str, Any]]] = []
    for cit in doc_citations:
        sid = str(cit.get("doc_id"))
        target_p = cit.get("page") or doc_page_map.get(sid, 1)
        with contextlib.suppress(ValueError, TypeError):
            target_p = int(target_p)
        targets.append((target_p, cit))
    targets.sort(key=lambda item: item[0])
    for target_p, cit in targets:
        sid = str(cit.get("doc_id"))

        # 1) 의미 없는 플레이스홀더 섹션명을 빈 문자열("")로 정규화
        raw_sec = (cit.get("section") or "").strip()
        is_generic = not raw_sec or raw_sec.lower() in [
            "general content",
            "none",
            "?",
            "문서 본문",
            "일반 본문",
        ]
        clean_sec = (
            ""
            if is_generic
            else re.sub(
                r"[\(\[]\s*p(?:age)?\.?\s*\d+\s*[\)\]]",
                "",
                raw_sec,
                flags=re.IGNORECASE,
            ).strip()
        )

        span = cit.get("text_span")
        norm = _normalize_text_for_dedup(span) if span else doc_text_map.get(sid, "")
        span_raw = span if isinstance(span, str) and span else doc_raw_map.get(sid, "")
        if target_p not in page_to_excerpt and span_raw:
            page_to_excerpt[target_p] = span_raw
        if norm:
            text_hash = fast_hash(norm)
            if text_hash in seen_text_hashes:
                kept = seen_text_hashes[text_hash]
                prev_sec = page_to_refs[kept]["section"]
                cand_sec = clean_sec or raw_sec
                if cand_sec and len(cand_sec) > len(prev_sec):
                    page_to_refs[kept]["section"] = cand_sec
                continue
            seen_text_hashes[text_hash] = target_p

        if target_p not in page_to_refs:
            page_to_refs[target_p] = {"sid": sid, "section": clean_sec}
        elif not page_to_refs[target_p]["section"] and clean_sec:
            # 더 구체적인 실제 섹션명이 있는 경우에만 갱신
            page_to_refs[target_p]["section"] = clean_sec

    # 4. 페이지 번호 오름차순 정렬
    sorted_pages = sorted(page_to_refs.keys())
    if not sorted_pages:
        return False

    # 5. UI 렌더링 (단일 페이지당 1개의 정제된 버튼 배치)
    for idx, page_num in enumerate(sorted_pages):
        ref_info = page_to_refs[page_num]
        sec_title = ref_info["section"]
        sid = ref_info["sid"]

        # 유의미한 섹션명이 존재할 때만 결합, 없을 때는 "p.X"로 단일화하여 중복 수식어 제거
        label = f"{sec_title} · p.{page_num}" if sec_title else f"p.{page_num}"

        if generating:
            # 생성 중 인용은 위젯 없이 렌더한다: 타임라인 본문과 aux
            # 익스팬더가 같은 턴을 동시에 렌더하므로, 버튼(안정 키든 자동
            # 키든)은 StreamlitDuplicateElement(Key|Id)로 스트리밍을 즉시
            # 깨뜨린다. span은 위젯 identity가 없어 안전하고, 버튼 크롬이
            # 없어 클릭 가능처럼 보이지도 않는다.
            st.markdown(
                f'{idx + 1}. <span data-doc-id="{html.escape(sid)}">'
                f"{html.escape(label)}</span>",
                unsafe_allow_html=True,
            )
        else:
            excerpt = normalize_excerpt(page_to_excerpt.get(page_num, ""))
            base_help = f"PDF {page_num}페이지로 이동"
            st.button(
                f"{label}",
                key=f"pop_doc_{msg_id}_{page_num}_{idx}",
                use_container_width=True,
                on_click=_handle_page_jump,
                args=(page_num,),
                help=f"{base_help} — {excerpt}" if excerpt else base_help,
            )
    return True


# [수정 후: src/ui/components/chat_references.py]
def render_generation_expander(
    msg: dict[str, Any],
    *,
    expanded: bool,
    generating: bool,
    status_text: str = "Answer generation",
) -> None:
    documents = msg.get("documents") or []
    citations = msg.get("citations") or []
    thought = (msg.get("thought") or "").strip()
    msg_id = msg.get("msg_id") or ""
    has_references = bool(documents or citations)

    if not thought and not has_references:
        return

    both_present = bool(has_references and thought)

    # 상단 익스팬더 제목 결정
    if both_present:
        expander_title = t("expander_sources_reasoning")  # "출처 및 추론 과정"
    elif has_references:
        expander_title = t("expander_sources_only")  # "인용 출처"
    else:
        expander_title = t("expander_reasoning_only")  # "사고 과정"

    auto_expand = expanded or (generating and bool(thought))

    with st.expander(
        expander_title,
        expanded=auto_expand,
        key=(f"gen_exp_{msg_id}" if msg_id and not generating else None),
    ):
        # 1. 사고 과정 본문 (인용구 인용 스타일로 표기하므로 중복 캡션 제거)
        if thought:
            st.markdown(
                "\n".join(f"> {line}" for line in thought.splitlines())
                if "\n" in thought
                else f"> {thought}"
            )

        # 2. 두 영역이 모두 있을 때만 경계 구분선 추가
        if both_present:
            st.divider()

        # 3. 출처 버튼 목록 렌더링 (show_caption=False로 전달하여 상단 제목과의 중복 캡션 제거)
        if has_references:
            _render_references_content(
                msg_id,
                documents,
                citations=citations,
                generating=generating,
                show_caption=False,  # <-- 중복 캡션 비활성화
            )
