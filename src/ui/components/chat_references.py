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
import logging
import re
import time
from typing import Any

import streamlit as st

from common.utils import doc_stable_id, fast_hash
from core.session import SessionManager
from ui.components.common import get_doc_metadata, navigate_to_page
from ui.strings import t

logger = logging.getLogger(__name__)

__all__ = [
    "_extract_reference_pages",
    "_handle_doc_jump",
    "_handle_page_jump",
    "_render_references_content",
    "is_ghost_state",
    "normalize_excerpt",
    "ordered_reference_entries",
    "render_generation_expander",
    "render_inline_citation_badges",
    "render_reference_list",
]


def _strip_markdown_for_label(text: str) -> str:
    """행 라벨용 마크다운 걷어내기: 헤더 마커·불릿·표 파이프·강조·링크 정리.

    청크 원문 앞부분이 발췌원이므로 ``#``/``##`` 섹션 마커나 표 ``|`` 구분자가
    행에 그대로 노출된다(스크린샷 P1). 단어 내 단일 ``_``(snake_case 식별자
    보호)는 유지하고, 짝을 이루는 강조 마커만 제거한다.
    """
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    cleaned = re.sub(r"(?m)^\s*[-•·]\s+", "", cleaned)
    cleaned = cleaned.replace("![", "[")
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"[*`~]", "", cleaned)
    # bare 대괄호 제거: 행 라벨의 :blue[] 색 래퍼 파싱을 깨뜨리므로.
    cleaned = re.sub(r"[\[\]]", "", cleaned)
    return cleaned.replace("|", " ")


def normalize_excerpt(text: object, max_chars: int = 500) -> str:
    """호버용 발췌 정규화: 마크다운 정리 + strip + 공백 축소 + ~500자 절단.

    Streamlit ``help=`` 툴팁은 ``\\n\\n``에서 깨지므로(#13339) 모든 개행을
    제거해 단일 행으로 만든다. 배지/점프 버튼 렌더 경로가 공유한다.
    보이는 잘림은 익스팬더 행의 1줄 캡션에 위임하므로, 툴팁에는 실제 발췌를
    최대한 담는다 (CHI hover>click). ``render_reference_list`` 행 라벨도
    공유하므로 헤더/표 마커가 행에 노출되지 않는다.
    """
    if not isinstance(text, str):
        return ""
    collapsed = re.sub(r"\s+", " ", _strip_markdown_for_label(text)).strip()
    if len(collapsed) > max_chars:
        # 절단 경계의 마침표 런을 접은 뒤 ...付与: 발췌원 trailing "..."와
        # 겹쳐 "...." 이중 말줄임이 되지 않도록 한다 (P5).
        cut = re.sub(r"[.…\s]+$", "", collapsed[:max_chars].rstrip())
        return (cut if cut else collapsed[:max_chars].rstrip()) + "..."
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
    """답변 말미 인용 표식을 한 줄의 테마 적응형 텍스트로 렌더한다.

    ``[1] [2] ...`` 형태의 단일 ``st.markdown`` 호출이며, 배지 위젯이나
    인라인 ``help=``/툴팁을 달지 않는다 — 표식은 익스팬더 행으로의 매핑
    앵커일 뿐이고, 호버 미리보기는 익스팬더 점프 버튼이 전담한다.
    번호는 ``ordered_reference_entries`` 단일 소스에서 오므로 익스팬더의
    페이지 정렬·병합 결과와 개수가 항상 일치한다.
    """
    entries = ordered_reference_entries(documents, citations)
    if not entries:
        return False
    markers = " ".join(f"[{idx + 1}]" for idx in range(len(entries)))
    st.markdown(markers)
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


def _coerce_page(value: object) -> int | None:
    """메타데이터/인용의 페이지 값을 1 이상 정수로 정규화한다."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str):
        try:
            page = int(value.strip())
        except ValueError:
            return None
        return page if page >= 1 else None
    return None


def render_reference_list(
    msg_id: str,
    documents: list[Any] | None,
    citations: list[dict[str, Any]] | None = None,
    max_rows: int = 5,
    excerpt_chars: int = 80,
) -> bool:
    """문서별 한 줄 요약 링크 목록: Reference見出 + tertiary 점프 버튼.

    문서 입력 순서(관련도 순)대로 문서별 최대 1행,
    ``"{i}. :blue[{요약}] · p.{N}"`` tertiary 버튼으로 렌더한다. 내용은 파랑,
    번호·페이지는 기본색으로 구분된다 (버튼 라벨 StreamlitMarkdown 렌더).
    발췌는 해당 문서의 citation ``text_span`` 우선, 없으면 ``page_content``
    앞부분이며 ``normalize_excerpt``로 공백 축소 + ``...`` 절단한다. 클릭은
    기존 ``_handle_page_jump`` 콜백 경로를 그대로 사용해
    ``pdf_target_page(manual)`` 토큰을 세팅한다 (P0 콜백 계약 유지).
    """

    def _doc_text(doc: Any) -> str:
        content = getattr(doc, "page_content", None)
        if content is None and isinstance(doc, dict):
            content = doc.get("page_content")
        return content if isinstance(content, str) else ""

    span_by_doc: dict[str, str] = {}
    page_by_doc: dict[str, int] = {}
    for cit in citations or []:
        if not isinstance(cit, dict):
            continue
        doc_id = cit.get("doc_id")
        if doc_id is None:
            continue
        key = str(doc_id)
        span = cit.get("text_span")
        if key not in span_by_doc and isinstance(span, str) and span.strip():
            span_by_doc[key] = span
        page = _coerce_page(cit.get("page"))
        if key not in page_by_doc and page is not None:
            page_by_doc[key] = page

    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for doc in documents or []:
        sid = doc_stable_id(doc)
        if sid in seen:
            continue
        seen.add(sid)
        excerpt = normalize_excerpt(
            span_by_doc.get(sid) or _doc_text(doc), excerpt_chars
        )
        if not excerpt:
            continue
        page = page_by_doc.get(sid)
        if page is None:
            page = _coerce_page(get_doc_metadata(doc).get("page")) or 1
        rows.append((excerpt, page))
        if len(rows) >= max(1, max_rows):
            break
    if not rows:
        return False
    st.markdown(f"**{t('ref_section_title')}**")
    for idx, (excerpt, page) in enumerate(rows, start=1):
        st.button(
            f"{idx}. :blue[{excerpt}] · p.{page}",
            key=f"reflink_{msg_id}_{idx}_{page}",
            type="tertiary",
            width="stretch",
            on_click=_handle_page_jump,
            args=(page,),
        )
    return True


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


def is_ghost_state(
    documents: list[Any] | None,
    citations: list[dict[str, Any]] | None,
) -> bool:
    """Ghost 조건: sources 문서 도착 + 인용 미도착 (스켈레톤 선행 렌더)."""
    return bool(documents) and not citations


def ordered_reference_entries(
    documents: list[Any] | None,
    citations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """인용 행의 순서 단일 소스 (익스팬더 행 + 말미 배지가 공유).

    페이지 번호(target_p) 및 대표 섹션 기준 그룹화, Branch-D 텍스트
    dedup(동일 정규화 텍스트는 최소 페이지만 유지, 가장 구체적인 섹션명
    보존), 페이지 오름차순 정렬을 수행한다. 인용이 없고 문서만 있으면
    섹션·페이지 기준 폴백 인용을 만든다. 반환은
    ``[{"page", "section", "sid", "excerpt"}]`` 리스트이며 순수 함수다
    (Streamlit 호출 없음 — 단위 테스트 직접 검증 가능).
    """
    # 1. 인용 데이터 수집
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
        return []

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

    # 3. 페이지 번호(target_p) 및 대표 섹션 기준 그룹화 (중복 버튼 방지)
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
    return [
        {
            "page": page_num,
            "section": page_to_refs[page_num]["section"],
            "sid": page_to_refs[page_num]["sid"],
            "excerpt": page_to_excerpt.get(page_num, ""),
        }
        for page_num in sorted(page_to_refs.keys())
    ]


def _render_references_content(
    msg_id: str,
    documents: list[Any] | None,
    citations: list[dict[str, Any]] | None = None,
    generating: bool = False,
    show_caption: bool = True,
) -> bool:
    _t_recompute = time.perf_counter()
    _n_recompute_docs = len(documents or [])
    _n_recompute_cits = len(citations or [])

    def _log_recompute() -> None:
        logger.info(
            "[PERF] recompute took %.3fs (%d docs, %d citations)",
            time.perf_counter() - _t_recompute,
            _n_recompute_docs,
            _n_recompute_cits,
        )

    # Ghost 스켈레톤: 생성 중 + 문서 도착 + 인용 미도착이면 위젯 없는
    # 플레이스홀더 행을 선행 렌더한다. 최종 렌더가 동일 msg_id 키로
    # 1:1 교체하므로 중복되지 않고, 인용이 바뀌면(위 조건 불충족) 즉시
    # 실경로로 전환되어 스테일 고스트가 남지 않는다. 실경로는 아래 그대로.
    if generating and is_ghost_state(documents, citations):
        for doc in documents or []:
            page = get_doc_metadata(doc).get("page", 1)
            with contextlib.suppress(ValueError, TypeError):
                page = int(page)
            st.markdown(
                f'<div data-ghost="true" data-msg-id="{html.escape(msg_id)}">'
                f"{html.escape(t('ref_ghost_searching').format(page=page))}</div>",
                unsafe_allow_html=True,
            )
        logger.info(
            "[GHOST] ghost rendered msg_id=%s (%d docs)",
            msg_id,
            _n_recompute_docs,
        )
        _log_recompute()
        return True

    rendered = False
    if not documents and not citations:
        _log_recompute()
        return rendered

    # 순서 단일 소스: 배지 번호와 익스팬더 행이 같은 entries를 공유한다.
    entries = ordered_reference_entries(documents, citations)
    if not entries:
        _log_recompute()
        return False

    # 5. UI 렌더링 (단일 페이지당 1개의 정제된 버튼 배치)
    for idx, entry in enumerate(entries):
        page_num = entry["page"]
        sec_title = entry["section"]
        sid = entry["sid"]

        # 유의미한 섹션명이 존재할 때만 결합, 없을 때는 "p.X"로 단일화하여 중복 수식어 제거
        label = f"{sec_title} · p.{page_num}" if sec_title else f"p.{page_num}"

        if generating:
            # 생성 중 인용은 위젯 없이 렌더한다: 타임라인 본문과 aux
            # 익스팬더가 같은 턴을 동시에 렌더하므로, 버튼(안정 키든 자동
            # 키든)은 StreamlitDuplicateElement(Key|Id)로 스트리밍을 즉시
            # 깨뜨린다. span은 위젯 identity가 없어 안전하고, 버튼 크롬이
            # 없어 클릭 가능처럼 보이지도 않는다. 발췌는 텍스트로 함께 두어
            # 스크린리더/Ctrl+F 접근을 유지한다 (위젯이 아니라 중복 위험 없음).
            excerpt_inline = normalize_excerpt(entry["excerpt"], 200)
            st.markdown(
                f'{idx + 1}. <span data-doc-id="{html.escape(sid)}">'
                f"{html.escape(label)}</span>"
                + (f" — {html.escape(excerpt_inline)}" if excerpt_inline else ""),
                unsafe_allow_html=True,
            )
        else:
            excerpt = normalize_excerpt(entry["excerpt"])
            base_help = t("ref_jump_help").format(page=page_num)
            st.button(
                f"{label}",
                key=f"pop_doc_{msg_id}_{page_num}_{idx}",
                width="stretch",
                on_click=_handle_page_jump,
                args=(page_num,),
                help=f"{base_help} — {excerpt}" if excerpt else base_help,
            )
            if excerpt and show_caption:
                st.caption(excerpt)
    _log_recompute()
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

        # 3. 출처 버튼 목록 렌더링. 완료 메시지는 발췌 캡션을 함께 보여
        # 스크린리더/Ctrl+F 접근을 유지하고, 생성 중에는 캡션을 생략해
        # 스트리밍 본문과의 중복을 피한다.
        if has_references:
            _render_references_content(
                msg_id,
                documents,
                citations=citations,
                generating=generating,
                show_caption=not generating,
            )
