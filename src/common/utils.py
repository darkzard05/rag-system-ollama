"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
Utils Rebuild: 복잡한 데코레이터 제거 및 비동기 헬퍼 단순화.
"""

import hashlib
import html
import logging
import os
import re
from collections.abc import Awaitable
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# --- 사전 컴파일된 정규표현식 (성능 최적화) ---
_RE_LATEX_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_RE_LATEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

# [수정] 복합 인용 패턴 지원: [1], [p.5], (page 10), [섹션: ..., p.5], [DOC 1, p.5] 등 광범위한 패턴 지원
_RE_CITATION_BLOCK = re.compile(
    r"(\[|(?:\s|^)\()((?:[^\]\)]*?[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*)(?:(?:[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*))*)([\]\)]|(?:\s|$))",
    re.IGNORECASE,
)
# [수정] 문서 인용 패턴: [doc:<stable_id>] — stable_id는 문서의 doc_id 메타데이터
# (없으면 page_content 해시)이며, enumerate 위치/페이지 번호가 아님. 기존
# _RE_CITATION_BLOCK이 [doc:chunk_B]를 페이지로 오인하는 버그를 막기 위해 전용
# 패턴으로 먼저 처리한다.
_RE_DOC_CITATION = re.compile(r"\[doc:([\w-]+)\]")

_RE_EXTRACT_PAGES = re.compile(r"(\d+)")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_CLEAN_LIST_NUM = re.compile(r"^\d+[\.\)]\s*")
_RE_CLEAN_LIST_BULLET = re.compile(r"^[\-\*•]\s*")

# [수정] 정규식 완화:
# 1. ^\d+[\.\)\s]+ : 문두의 숫자와 점/괄호 (예: "1. ", "1) ")
# 2. ^\s*[\-\*\u2022]\s* : 문두의 불렛 포인트 (예: "- ", "* ")
# 3. ^["']+|["']+$ : 문두/문미의 따옴표
# 4. (?:^Example:|^Query:)\s* : "Example:" 같은 접두사 제거
_RE_QUERY_CLEAN_PREFIX = re.compile(
    r"^(?:\d+[\.\)\s]+|\s*[\-\*\u2022]\s*|(?:Example|Query|Question):\s*)+",
    re.IGNORECASE,
)
_RE_QUERY_CLEAN_QUOTES = re.compile(r'^["\']+|["\']+$')


_RE_CODE_BLOCK = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")


def normalize_latex_delimiters(text: str) -> str:
    r"""
    LLM이 출력하는 다양한 LaTeX 수식 구분자를 Streamlit 표준($ 또는 $$)으로 변환합니다.
    - \( ... \) -> $ ... $ (인라인)
    - \[ ... \] -> $$ ... $$ (블록)
    - 기호 앞뒤의 불필요한 이스케이프 제거
    - 코드 블록 내의 내용은 변환에서 제외하여 코드 예제 보호
    """
    if not text:
        return text

    # 0. 코드 블록(```...``` / `...`) 임시 치환 — 내부 LaTeX 변환 방지
    code_blocks: list[str] = []

    def _save_code(m: re.Match[str]) -> str:
        code_blocks.append(m.group(0))
        return f"\x00LATEX_BLOCK_{len(code_blocks) - 1}\x00"

    text = _RE_CODE_BLOCK.sub(_save_code, text)

    # 1. 블록 수식 변환: \[ ... \] -> $$ ... $$
    text = _RE_LATEX_BLOCK.sub(r"$$\1$$", text)

    # 2. 인라인 수식 변환: \( ... \) -> $ ... $
    text = _RE_LATEX_INLINE.sub(r"$\1$", text)

    # 3. 잘못된 이스케이프 문자 정제 (예: \$ -> $)
    text = text.replace(r"\$", "$")

    # 4. 코드 블록 복원
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00LATEX_BLOCK_{i}\x00", block)

    return text


def _build_line_boxes(coords: list, page_val: int, content: str) -> list[dict]:
    """단일 페이지 좌표에서 줄 단위 하이라이트 박스 annotation 목록 생성."""
    # [고도화] 연속성 기반 텍스트 매칭 (Sequence Matching)
    # 1. 청크 텍스트와 PDF 텍스트를 순수 단어 토큰으로 정규화
    content_tokens = re.findall(r"[\w\d]+", content)
    if not content_tokens:
        return []

    pdf_tokens = [re.sub(r"[^\w\d]", "", str(c[4]).lower()) for c in coords]

    # 2. PDF 단어 리스트에서 현재 청크가 시작되는 최적의 지점 검색 (Sliding Window)
    best_start = -1
    max_match = 0
    window_size = min(20, len(content_tokens))  # 시작 부분 20단어로 지점 탐색

    for j in range(len(pdf_tokens) - len(content_tokens) + 1):
        current_match = 0
        for k in range(window_size):
            if pdf_tokens[j + k] == content_tokens[k]:
                current_match += 1

        if current_match > max_match:
            max_match = current_match
            best_start = j

        # 80% 이상 일치하면 즉시 시작점으로 확정 (성능 최적화)
        if current_match >= window_size * 0.8:
            best_start = j
            break

    # 3. 매칭된 지점부터 청크 길이만큼의 좌표만 추출
    if best_start != -1:
        # 청크 텍스트 내의 실제 단어 개수만큼 좌표를 가져옴
        filtered_coords = coords[best_start : best_start + len(content_tokens)]
    else:
        # 매칭 실패 시에만 기존의 루즈한 필터링으로 폴백 (최소 가시성)
        filtered_coords = [
            c
            for c in coords
            if re.sub(r"[^\w\d]", "", str(c[4]).lower()) in content_tokens[:50]
        ]

    if not filtered_coords:
        return []

    # 4. 줄 단위 그룹화 및 박스 생성
    lines: dict[int, list] = {}
    for c in filtered_coords:
        y_key = round(c[1] / 8) * 8
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append(c)

    annotations: list[dict] = []
    for y_key in sorted(lines.keys()):
        line_coords = lines[y_key]
        x_min = min(c[0] for c in line_coords)
        y_min = min(c[1] for c in line_coords)
        x_max = max(c[2] for c in line_coords)
        y_max = max(c[3] for c in line_coords)

        if x_max > x_min and y_max > y_min:
            annotations.append(
                {
                    "page": page_val,
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                    "color": "red",
                }
            )

    if annotations:
        logger.debug(
            f"[HIGHLIGHT] Page {page_val}: Found chunk sequence at index {best_start}, created {len(annotations)} line boxes"
        )

    return annotations


def extract_annotations_from_docs(documents: list) -> list[dict]:
    """
    검색된 문서들의 메타데이터에서 좌표 정보를 추출하여
    현재 청크의 텍스트와 일치하는 영역만 줄(Line) 단위로 하이라이트합니다.
    """
    annotations: list[dict] = []
    if not documents:
        return annotations

    logger.info(f"[HIGHLIGHT] Processing {len(documents)} docs for annotations")

    for _i, doc in enumerate(documents):
        meta = (
            getattr(doc, "metadata", {})
            if hasattr(doc, "metadata")
            else doc.get("metadata", {})
        )
        content = (
            getattr(doc, "page_content", "")
            if hasattr(doc, "page_content")
            else doc.get("page_content", "")
        ).lower()
        file_path = meta.get("file_path") or meta.get("source")

        # 다중 페이지 청크: 페이지별 좌표(page_coords)로 줄 단위 하이라이트 생성
        page_coords = meta.get("page_coords")  # dict[int, list] | None
        if page_coords:
            for page_val, coords in sorted(page_coords.items()):
                if coords:
                    annotations.extend(
                        _build_line_boxes(coords, int(page_val), content)
                    )
        else:
            # 기존 단일 페이지 경로 (word_coords 또는 on-demand fitz)
            page_val = int(meta.get("page", 1))
            all_coords = meta.get("word_coords", [])

            # [고도화] On-demand 좌표 추출 (Strategy C)
            # 메타데이터에 좌표가 없으면 PDF 파일에서 실시간으로 검색합니다.
            if not all_coords and file_path and os.path.exists(file_path):
                try:
                    import pymupdf as fitz

                    with fitz.open(file_path) as pdf:
                        page = pdf[page_val - 1]
                        textpage = page.get_textpage()

                        # [고도화] 개선된 검색 쿼리 전처리 로직 (Strategy C 기반)
                        # 1. HTML 태그 제거 (예: <img src="...">)
                        text = re.sub(r"<[^>]+>", " ", content)

                        # 2. 마크다운 및 특수문자 제거 (기존보다 강화, 따옴표 포함)
                        text = re.sub(r"[#*`_~\[\]()\"']", "", text)

                        # 3. 연속 공백 제거 및 앞뒤 공백 제거
                        text = re.sub(r"\s+", " ", text).strip()

                        # 4. 소문자 변환
                        clean_content = text.lower()

                        # 5. 문장 분리 (줄바꿈 포함)
                        raw_sentences = re.split(r"[.!?\n]", clean_content)

                        sentences = []
                        for s in raw_sentences:
                            s = s.strip()

                            # [필터링 1] 최소 길이 상향 (8 -> 20)
                            # 너무 짧은 문장은 오탐지(False Positive)의 원인이 됨
                            if len(s) < 20:
                                continue

                            # [필터링 2] 숫자나 특수문자로만 구성된 쓰레기 데이터 제거
                            if re.match(r"^[\d\s\W]+$", s):
                                continue

                            # [필터링 3] 표/그림 캡션 등 불필요한 메타데이터 제거 (휴리스틱)
                            # 예: "table 1", "figure 3", "page 10" 등으로 시작하는 경우
                            if re.match(r"^(table|figure|fig\.|tab\.)\s*\d+", s):
                                continue

                            # [필터링 4] 참고문헌 패턴 (예: [1], (2020)) 등으로 시작하는 경우
                            if re.match(r"^[\(\[]\s*\d+\s*[\)\]]", s):
                                continue

                            sentences.append(s)

                        if not sentences and clean_content:
                            # 폴백: 필터링 결과가 없으면 첫 150자 사용
                            sentences = [clean_content[:150].strip()]

                        doc_quads = []
                        # [최적화] TextPage를 사용하여 고속 검색
                        for search_query in sentences:
                            if not search_query:
                                continue

                            logger.info(
                                f"[HIGHLIGHT] Searching query on page {page_val}: '{search_query}'"
                            )

                            # [개선] 긴 문장 검색은 실패 확률이 높으므로 40자씩 끊어서 검색 (Overlapping Search)
                            chunk_len = 40
                            overlap = 10
                            for i in range(0, len(search_query), chunk_len - overlap):
                                part = search_query[i : i + chunk_len].strip()
                                if len(part) < 12:
                                    continue
                                quads = page.search_for(part, textpage=textpage)
                                if quads:
                                    doc_quads.extend(quads)

                        if doc_quads:
                            # [핵심 개선] 줄 단위 병합 로직 (Line Merging)
                            on_demand_lines: dict[float, list] = {}
                            for q in doc_quads:
                                y_key = round(q.y0 / 5) * 5
                                if y_key not in on_demand_lines:
                                    on_demand_lines[y_key] = []
                                on_demand_lines[y_key].append(q)

                            for y_key in sorted(on_demand_lines.keys()):
                                group = on_demand_lines[y_key]
                                x_min = min(r.x0 for r in group)
                                y_min = min(r.y0 for r in group)
                                x_max = max(r.x1 for r in group)
                                y_max = max(r.y1 for r in group)

                                annotations.append(
                                    {
                                        "page": page_val,
                                        "x": x_min,
                                        "y": y_min,
                                        "width": x_max - x_min,
                                        "height": y_max - y_min,
                                        "color": "red",
                                    }
                                )
                            continue
                except Exception as e:
                    logger.error(f"[HIGHLIGHT] On-demand search failed: {e}")
            else:
                logger.debug(
                    f"[HIGHLIGHT] Conditions not met: coords={len(all_coords)}, path_exists={os.path.exists(file_path) if file_path else 'N/A'}"
                )

            if not all_coords:
                continue

            annotations.extend(_build_line_boxes(all_coords, page_val, content))

    if annotations:
        pages = sorted({a["page"] for a in annotations})
        logger.info(
            f"[HIGHLIGHT] PDF 하이라이트 생성 완료: {len(annotations)}개 영역 (대상 페이지: {pages})"
        )

    return annotations


def apply_tooltips_to_response(
    response_text: str,
    documents: list | None = None,
    msg_index: int = 0,
    citations: list[dict] | None = None,
) -> str:
    """
    답변 내의 인용구([1], [p.5] 등)에 문서 정보 툴팁을 입힙니다.
    스팬은 시각적 구분용이며 클릭 동작이 없습니다 (단순 메타데이터).
    실제 페이지 이동은 네이티브 참조 popover 버튼이 담당합니다.

    citations: 구조화 인용 배열(doc_id 기반). 있으면 본문 말미에
    data-doc-id 앵커 소스 블록을 덧붙여 안정적 doc 점프를 지원합니다.
    """
    if not response_text:
        return response_text

    # 1. LaTeX 정규화 먼저 수행
    text = normalize_latex_delimiters(response_text)

    if not documents:
        return text

    # [doc:<stable_id>] -> 문서를 stable id(doc_id 또는 content 해시)로 조회.
    # 위치 기반 인덱스가 아니라 멤버십 조회하므로 rerank 재정렬 후에도
    # 동일 청크를 가리킨다 (P2: 안정 ID 매핑). 못 찾으면 원본 그대로 반환.
    doc_by_stable_id: dict[str, Any] = {}
    for _doc in documents:
        if hasattr(_doc, "metadata"):
            _meta = _doc.metadata or {}
            _content = getattr(_doc, "page_content", "") or ""
        else:
            _meta = _doc.get("metadata", {}) or {}
            _content = _doc.get("page_content", "") or ""
        _sid = (
            str(_meta.get("doc_id"))
            if _meta.get("doc_id") is not None
            else fast_hash(_content)
        )
        doc_by_stable_id[_sid] = _doc

    def replace_doc_citation(match):
        # [doc:<stable_id>] — stable_id는 page_content 해시/위치가 아니라
        # 문서 식별자. 페이지 정규식이 stable_id를 페이지로 오인하는 버그를
        # 차단하기 위해 전용 패턴이 먼저 처리한다.
        cited_id = match.group(1)
        if cited_id not in doc_by_stable_id:
            # 알 수 없는 id면 원본 그대로 반환 (죽은 인용 방지).
            return match.group(0)

        doc = doc_by_stable_id[cited_id]
        content = (
            getattr(doc, "page_content", "")
            if hasattr(doc, "page_content")
            else doc.get("page_content", "")
        )
        clean_content = html.escape(content).replace("\n", " ").strip()[:300] + "..."

        return (
            f'<span class="citation-highlight" title="{clean_content}" '
            f'data-doc-id="{cited_id}" '
            f'style="color: #007bff; font-weight: 600; text-decoration: underline; text-underline-offset: 3px;">'
            f"{match.group(0)}</span>"
        )

    def replace_citation(match):
        full_match = match.group(0).strip()
        inner_text = match.group(2)

        # 1. 페이지 번호 추출 (p.X 또는 page X 패턴 우선 검색)
        target_page = -1
        p_match = re.search(r"[Pp](?:age)?\.?\s*(\d+)", inner_text)
        if p_match:
            target_page = int(p_match.group(1))
        else:
            # 키워드가 없는 경우 첫 번째 숫자 시도
            page_matches = _RE_EXTRACT_PAGES.findall(inner_text)
            if page_matches:
                target_page = int(page_matches[0])

        if target_page == -1:
            return full_match

        # 2. 섹션명 추출 시도 (예: "[섹션: 3 CM3, p.3]" -> "3 CM3")
        target_section = None
        if "섹션:" in inner_text:
            try:
                # '섹션:' 이후부터 ',' 또는 'p.' 이전까지 추출
                sec_part = inner_text.split("섹션:")[1]
                target_section = sec_part.split(",")[0].strip()
            except Exception:
                pass

        # 3. 툴팁에 표시할 문서 내용 찾기
        clean_content = "인용된 원문 정보를 불러올 수 없습니다."
        best_doc = None

        # [최적화] 섹션명과 페이지가 모두 일치하는 문서를 최우선으로 찾음
        for doc in documents:
            meta = (
                getattr(doc, "metadata", {})
                if hasattr(doc, "metadata")
                else doc.get("metadata", {})
            )
            doc_page = int(meta.get("page", -1))
            doc_section = meta.get("current_section", "")

            if doc_page == target_page:
                if target_section and target_section in doc_section:
                    best_doc = doc
                    break
                if not best_doc:  # 일단 페이지라도 맞으면 후보로 등록
                    best_doc = doc

        if best_doc:
            content = (
                getattr(best_doc, "page_content", "")
                if hasattr(best_doc, "page_content")
                else best_doc.get("page_content", "")
            )
            # HTML title 속성에 넣기 위해 이스케이프 (XSS 방어)
            clean_content = (
                html.escape(content).replace("\n", " ").strip()[:300] + "..."
            )

        # 인용 스타일: 색/밑줄로 시각적 구분만 유지한다. 스팬은 클릭
        # 동작이 없으므로 cursor: pointer를 사용하지 않는다 (죽은 제스처 방지).
        # 페이지 이동은 네이티브 참조 popover 버튼(documents 기반)이 담당.
        # data-page는 순수 메타데이터로만 남긴다 (클릭 트리거가 아님).
        return (
            f'<span class="citation-highlight" title="{clean_content}" '
            f'data-page="{target_page}" '
            f'style="color: #007bff; font-weight: 600; text-decoration: underline; text-underline-offset: 3px;">'
            f"{full_match}</span>"
        )

    try:
        # 1. 문서 인용 [doc:N]을 먼저 위치 기반으로 치환 (페이지 오인 방지).
        text = _RE_DOC_CITATION.sub(replace_doc_citation, text)
        # 2. 페이지 인용 [p.3] 등을 처리 ([doc:N]은 이미 변환되어 영향 없음).
        text = _RE_CITATION_BLOCK.sub(replace_citation, text)
    except Exception as e:
        logger.error(f"[Utils] 인용구 처리 오류: {e}")

    # 3. 구조화 citations[]를 본문 말미의 data-doc-id 소스 앵커로 덧붙인다.
    #    PRIMARY 소스는 citations[] (doc_id 기반, rerank 후에도 안정).
    #    인라인 [doc:N] 폴백은 위 단계에서 이미 처리된다.
    if citations:
        anchors: list[str] = []
        for idx, cit in enumerate(citations):
            if not isinstance(cit, dict):
                continue
            sid = cit.get("doc_id")
            if sid is None:
                continue
            label = html.escape(
                str(cit.get("text_span") or cit.get("section") or f"Source {idx + 1}")
            )[:160]
            anchors.append(
                f'<span class="citation-source" data-doc-id="{html.escape(str(sid))}" '
                f'style="color: #007bff; font-weight: 600; margin-right: 8px;">'
                f"[{idx + 1}] {label}</span>"
            )
        if anchors:
            text += (
                '\n\n<span class="citation-sources" data-doc-ids="{}">{}</span>'.format(
                    ",".join(
                        html.escape(str(c.get("doc_id")))
                        for c in citations
                        if isinstance(c, dict) and c.get("doc_id") is not None
                    ),
                    "".join(anchors),
                )
            )

    return text


# --- 전처리용 고속 테이블 ---
# 널 문자 등 제어 문자를 공백으로 치환하는 테이블
_CLEAN_TRANS_TABLE = str.maketrans({"\x00": " ", "\r": " ", "\n": " ", "\t": " "})


def preprocess_text(text: str) -> str:
    """
    텍스트 정제: 제어 문자를 공백으로 치환하고 연속 공백을 고속 정규화
    [최적화] 정규식 엔진 대신 네이티브 split/join을 사용하여 오버헤드 최소화
    """
    if not text:
        return ""

    # 1. str.translate를 이용한 고속 문자 치환
    text = text.translate(_CLEAN_TRANS_TABLE)

    # 2. 연속된 공백을 단일 공백으로 통합 (split/join이 re.sub보다 훨씬 빠름)
    return " ".join(text.split())


def safe_cache_data(func=None, **kwargs):
    """Streamlit 런타임이 있을 때만 cache_data를 적용하고, 없으면 원본 함수를 반환합니다."""
    if func is None:
        return lambda f: safe_cache_data(f, **kwargs)

    try:
        if st.runtime.exists():
            return st.cache_data(**kwargs)(func)
    except Exception:
        pass
    return func


def safe_cache_resource(func=None, **kwargs):
    """Streamlit 런타임이 있을 때만 cache_resource를 적용하고, 없으면 원본 함수를 반환합니다."""
    if func is None:
        return lambda f: safe_cache_resource(f, **kwargs)

    try:
        if st.runtime.exists():
            return st.cache_resource(**kwargs)(func)
    except Exception:
        pass
    return func


def fast_hash(text: str, length: int = 16) -> str:
    """
    보안이 필요 없는 단순 식별용 고속 해시 함수.
    SHA256보다 훨씬 빠른 MD5를 사용하고 결과 길이를 조절합니다.
    """
    if not text:
        return "0" * length
    if isinstance(text, bytes):
        text = text.decode(errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    # usedforsecurity=False: 보안 진단 도구(Bandit 등)에 이 해시가
    # 암호화나 보안 목적으로 사용되지 않음을 알립니다.
    return hashlib.md5(text.encode(errors="ignore"), usedforsecurity=False).hexdigest()[
        :length
    ]


def count_tokens_rough(text: str) -> int:
    """
    텍스트의 토큰 수를 대략적으로 계산합니다. (보수적 추정)
    - 영어/숫자/공백(ASCII): 약 3~4글자당 1토큰
    - 한글/특수문자(비ASCII): 약 1글자당 1.4토큰

    R4-02: Ollama `prompt_eval_count` 실측 대비 기존 비ASCII 가중치(2.5/문자)가
    2.32배 과대추정이었다(동일 프롬프트 추정 1388 vs 실제 598). 실측 보정 결과
    한글 1글자당 1.2~1.5토큰 수준이 실제와 근접하여, 안전 마진을 남긴 1.4로
    재보정했다. 한글/비ASCII가 ASCII보다 토큰당 문자 수가 적다는 보수적 원칙은 유지한다.
    """

    if not text:
        return 0

    # 1. ASCII 문자(영어, 숫자, 기본 기호, 공백) 개수 파악
    ascii_pattern = r"[a-zA-Z0-9\s.,!?;:()\[\]{}<>\-_=+\x00-\x7F]"
    ascii_chars = len(re.findall(ascii_pattern, text))

    # 2. 비ASCII(한글, 한자 등) 문자 개수 파악
    non_ascii_chars = len(text) - ascii_chars

    # 3. 보수적 가중치 적용 (ASCII는 3글자당 1토큰, 비ASCII는 1글자당 1.4토큰 — R4-02 보정)
    rough_count = (ascii_chars / 3.0) + (non_ascii_chars * 1.4)

    # 최소 1개 이상 반환 및 정수 올림 처리 효과
    return int(rough_count) + 1


def run_in_background_worker(coro: Awaitable[Any], session_id: str) -> None:
    """
    Streamlit 환경에서 코루틴을 AsyncWorker의 전용 이벤트 루프에서 실행하는 백그라운드 워커.
    - run_coroutine_threadsafe로 스레드 안전하게 코루틴 제출
    - 작업 완료 후 자동으로 rerun 트리거
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from common.async_worker import AsyncWorker
    from core.session import SessionManager

    ctx = get_script_run_ctx()

    async def _with_session() -> Any:
        SessionManager.set_session_id(session_id)
        return await coro

    def _on_complete(future):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Background worker error: {e}", exc_info=True)

        if ctx and ctx.session_id:
            try:
                from streamlit.runtime import get_instance

                runtime = get_instance()
                if runtime:
                    session_info = runtime._session_mgr.get_session_info(ctx.session_id)
                    if session_info:
                        session_info.session.request_rerun(None)
            except Exception as e:
                # rerun 재요청 실패 시 입력창이 영구 비활성화되지 않도록
                # 세션 플래그를 정리한다(INT-입력동결). 폴링 fragment가
                # 0.5초마다 상태를 재계산하므로 다음 폴링에서 복구된다.
                logger.error(f"Background worker rerun failed: {e}", exc_info=True)
                try:
                    from core.session import SessionManager

                    SessionManager.set_session_id(ctx.session_id)
                    SessionManager.set("is_building_rag", False, ctx.session_id)
                    SessionManager.set("is_swapping_model", False, ctx.session_id)
                    SessionManager.set("is_generating_answer", False, ctx.session_id)
                except Exception as inner:  # noqa: BLE001 - 복구 실패는 로그만
                    logger.error(f"Flag recovery failed: {inner}", exc_info=True)

    future = AsyncWorker().submit(_with_session())
    future.add_done_callback(_on_complete)
