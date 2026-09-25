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

from common.utils import doc_stable_id
from core.session import SessionManager
from ui.components.common import get_doc_metadata, navigate_to_page
from ui.strings import t

__all__ = [
    "_extract_reference_pages",
    "_handle_doc_jump",
    "_handle_page_jump",
    "_render_references_content",
    "render_generation_expander",
]


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

    # 3. [개선] 페이지 번호(target_p) 및 대표 섹션 기준 그룹화 (중복 버튼 방지)
    page_to_refs: dict[int, dict[str, Any]] = {}
    for cit in doc_citations:
        sid = str(cit.get("doc_id"))
        target_p = cit.get("page") or doc_page_map.get(sid, 1)
        with contextlib.suppress(ValueError, TypeError):
            target_p = int(target_p)

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
            st.button(
                f"{label}",
                key=f"pop_doc_{msg_id}_{page_num}_{idx}",
                use_container_width=True,
                on_click=_handle_page_jump,
                args=(page_num,),
                help=f"PDF {page_num}페이지로 이동",  # 장황한 설명 문구 간결화
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
