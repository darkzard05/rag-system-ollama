"""utils.py — 프로젝트 공용 유틸리티 단일 진원 (Phase 1A 병합).

구 도메인 유틸 5개 모듈을 흡수했다:
- 해시(``fast_hash``/``doc_stable_id``/``compute_file_hash``): 본 모듈 직접 정의
- Streamlit 캐시 래퍼(``safe_cache_data``/``safe_cache_resource``): 본 모듈 직접 정의
- PDF 주석/툴팁(``extract_annotations_from_docs`` 등): 본 모듈 직접 정의
- 텍스트 유틸: ``common.text_utils`` 가 진원, 본 모듈은 하위 호환 재수출
- 백그라운드 워커: ``common.async_worker`` 가 진원, 본 모듈은 하위 호환 재수출

구경로는 순수 re-export shim으로 유지한다
(Phase 1 only, Task10에서 제거).
"""

import hashlib
import html
import logging
import os
import re
from typing import Any

import streamlit as st

from common.async_worker import run_in_background_worker
from common.pdf_utils import open_pdf_document
from common.text_utils import (
    count_tokens_rough,
    normalize_latex_delimiters,
    preprocess_text,
    strip_context_tokens,
)

logger = logging.getLogger(__name__)


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


def doc_stable_id(
    doc: Any,
    *,
    page_content_key: str = "page_content",
) -> str:
    """문서의 안정 식별자(doc_id 메타 우선, 없으면 page_content 해시)를 반환합니다.

    ``Document`` 객체와 ``dict``(``{page_content, metadata}``) 를 모두 지원합니다.
    안정 키는 ``metadata.get("doc_id")`` 를 우선 사용하고, 없으면
    ``fast_hash(page_content)`` 로 계산합니다. 반환값은 항상 ``str`` 입니다.
    """
    if isinstance(doc, dict):
        meta = doc.get("metadata", {}) or {}
        content = doc.get(page_content_key, "") or ""
    else:
        meta = getattr(doc, "metadata", None) or {}
        content = getattr(doc, page_content_key, "") or ""
    doc_id = meta.get("doc_id")
    if doc_id is not None:
        return str(doc_id)
    return fast_hash(str(content))


def compute_file_hash(
    file_path: str,
    *,
    data: bytes | None = None,
    algorithm: str = "sha256",
) -> str:
    """파일 또는 바이트 데이터의 해시를 계산합니다. (R: Group 4 중복 통합)

    ``document_processor.compute_file_hash`` 와 ``cache_security`` 의
    ``compute_file_hash`` 를 단일 구현으로 통합한 공용 함수입니다.

    - ``data`` 가 주어지면 파일을 읽지 않고 해당 바이트를 해시합니다.
    - ``algorithm`` 은 ``hashlib.new`` 가 지원하는 알고리즘 이름입니다.
    - 두 기존 구현의 안전 계약을 모두 유지합니다:
      파일이 존재하지 않으면 ``FileNotFoundError`` 를, 그 외 I/O/해시 오류는
      ``""`` (빈 문자열) 을 반환합니다.
    """
    try:
        hasher = hashlib.new(algorithm)
        if data is not None:
            hasher.update(data)
            return hasher.hexdigest()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        raise
    except OSError as e:
        logger.error(f"파일 해시 계산 실패 ({file_path}): {e}")
        return ""


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


def _memo_key_for_doc(
    meta: dict[str, Any], page_val: int, content: str
) -> tuple[str, int, str] | None:
    """어노테이션 메모 키를 구성합니다 (file_hash+page+normtext 해시).

    ``file_hash`` 가 없으면 메모를 사용하지 않고 ``None`` 을 반환합니다.
    """
    file_hash = meta.get("file_hash")
    if not file_hash:
        return None
    try:
        from cache.coord_cache import memo_norm_hash
    except ImportError as e:
        logger.debug(f"[HIGHLIGHT] 메모 해시 로드 실패: {e}")
        return None
    return (str(file_hash), int(page_val), memo_norm_hash(content))


def _memo_lookup(key: tuple[str, int, str]) -> list[dict] | None:
    """메모 적중 시 라인박스를 반환합니다 (fitz 오픈 없음, never raise)."""
    try:
        from cache.coord_cache import coord_cache
    except ImportError as e:
        logger.debug(f"[HIGHLIGHT] 메모 조회 불가: {e}")
        return None
    try:
        return coord_cache.get_annotation_memo_sync(*key)
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.debug(f"[HIGHLIGHT] 메모 조회 실패: {e}")
        return None


def _memo_store(key: tuple[str, int, str], doc_annots: list[dict]) -> None:
    """확정 라인박스를 메모에 저장합니다 (실패해도 추출 결과에 영향 없음)."""
    try:
        from cache.coord_cache import coord_cache
    except ImportError as e:
        logger.debug(f"[HIGHLIGHT] 메모 저장 불가: {e}")
        return
    try:
        coord_cache.save_annotation_memo_sync(*key, doc_annots)
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.debug(f"[HIGHLIGHT] 메모 저장 실패: {e}")


def _ranked_doc_indices(
    documents: list, citations: list[dict[str, Any]] | None
) -> list[int]:
    """인용 순위대로 문서 인덱스를 정렬합니다 (Task 4 lazy prewarm).

    ``citations`` 가 없으면 입력 순서(검색 랭크) 그대로 반환합니다.
    있으면 ``doc_id``/``page`` 가 인용된 문서만 인용 등장 순서대로 반환하고,
    인용되지 않은 페이지는 제외합니다 (뷰어는 한 페이지만 보여주므로
    표시되지 않을 페이지는 eager 계산하지 않습니다). 인용과 매칭되는
    문서가 하나도 없으면(stale 인용) 전체를 입력 순서대로 반환합니다.
    """
    total = list(range(len(documents)))
    if not citations:
        return total
    doc_rank: dict[str, int] = {}
    page_rank: dict[int, int] = {}
    for order, cit in enumerate(citations):
        if not isinstance(cit, dict):
            continue
        sid = cit.get("doc_id")
        if sid is not None:
            key = str(sid)
            if key not in doc_rank:
                doc_rank[key] = order
        try:
            cited_page = int(cit["page"]) if cit.get("page") is not None else None
        except (TypeError, ValueError):
            cited_page = None
        if cited_page is not None and cited_page not in page_rank:
            page_rank[cited_page] = order
    if not doc_rank and not page_rank:
        return total
    scored: list[tuple[int, int]] = []
    for idx, doc in enumerate(documents):
        try:
            stable_id = doc_stable_id(doc)
        except (TypeError, ValueError, AttributeError):
            stable_id = ""
        if isinstance(doc, dict):
            meta = doc.get("metadata", {}) or {}
        else:
            meta = getattr(doc, "metadata", None) or {}
        try:
            doc_page = int(meta.get("page", 1))
        except (TypeError, ValueError):
            doc_page = 1
        best = min(
            doc_rank.get(stable_id, 10**9),
            doc_rank.get(str(meta.get("doc_id", "")), 10**9)
            if meta.get("doc_id") is not None
            else 10**9,
            page_rank.get(doc_page, 10**9),
        )
        if best >= 10**9:
            continue
        scored.append((best, idx))
    if not scored:
        return total
    scored.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in scored]


def _extract_search_sentences(content: str) -> list[str]:
    """하이라이트 검색 쿼리 문장을 추출합니다 (5 regex 패스 유지, 최대 3문장).

    호출자는 이미 소문자화된 본문을 넘기며, fitz 핸들 바깥에서 호출해도
    되지만 Task 4 그룹화 경로에서는 단일 오픈 블록 안에서 페이지별로
    호출됩니다 (오픈은 파일당 1회, 문장당 검색만 반복).
    """
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

    sentences: list[str] = []
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

    return sentences[:3]


def extract_annotations_from_docs(
    documents: list, citations: list[dict[str, Any]] | None = None
) -> list[dict]:
    """
    검색된 문서들의 메타데이터에서 좌표 정보를 추출하여
    현재 청크의 텍스트와 일치하는 영역만 줄(Line) 단위로 하이라이트합니다.

    Task 4 lazy citation-rank prewarm: 같은 PDF의 여러 청크는 파일당 1회만
    열고(``open_pdf_document`` 호이스트), ``citations`` 가 주어지면 인용된
    페이지부터 인용 순서대로 계산합니다. 표시되지 않을 페이지는 eager
    계산하지 않으며, 나머지는 참조 클릭 시점에 lazy 계산됩니다. 메모 적중은
    fitz 오픈 없이 O(1) 반환됩니다 (Wave2 계약 유지).
    """
    annotations: list[dict] = []
    if not documents:
        return annotations

    logger.info(f"[HIGHLIGHT] Processing {len(documents)} docs for annotations")

    ranked_indices = _ranked_doc_indices(documents, citations)
    pending: list[dict[str, Any]] = []

    for _i in ranked_indices:
        doc = documents[_i]
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

        # coord_cache_error: 좌표 캐시 읽기 실패로 좌표를 전혀 얻을 수 없음.
        # 일반 '좌표 없음'과 구분하기 위해 on-demand fitz 추출(일반 경로)을
        # 건너뛰어 이 문서는 빈 하이라이트로 남긴다. (실패를 명시적으로 표시)
        if meta.get("coord_cache_error"):
            logger.info(f"[HIGHLIGHT] coord_cache_error: 하이라이트 생략 ({file_path})")
            continue

        # 다중 페이지 청크: 페이지별 좌표(page_coords)로 줄 단위 하이라이트 생성
        page_coords = meta.get("page_coords")  # dict[int, list] | None
        if page_coords:
            for page_val, coords in sorted(page_coords.items()):
                if coords:
                    annotations.extend(
                        _build_line_boxes(coords, int(page_val), content)
                    )
            continue

        # 기존 단일 페이지 경로 (word_coords 또는 on-demand fitz)
        try:
            page_val = int(meta.get("page", 1))
        except (TypeError, ValueError):
            page_val = 1
        all_coords = meta.get("word_coords", [])
        if all_coords:
            annotations.extend(_build_line_boxes(all_coords, page_val, content))
            continue

        # [고도화] On-demand 좌표 추출 (Strategy C)
        # 메타데이터에 좌표가 없으면 PDF 파일에서 실시간으로 검색합니다.
        # [Wave2] 메모 우선: 적중이면 fitz 오픈 없이 O(1) 반환합니다.
        if not file_path or not os.path.exists(file_path):
            logger.debug(
                f"[HIGHLIGHT] Conditions not met: coords={len(all_coords)}, path_exists={os.path.exists(file_path) if file_path else 'N/A'}"
            )
            continue
        memo_key: tuple[str, int, str] | None = _memo_key_for_doc(
            meta, page_val, content
        )
        if memo_key is not None:
            memo_hit = _memo_lookup(memo_key)
            if memo_hit is not None:
                annotations.extend(memo_hit)
                continue
        pending.append(
            {
                "content": content,
                "file_path": str(file_path),
                "page_val": page_val,
                "memo_key": memo_key,
            }
        )

    # Task 4: 파일당 단일 오픈 — 같은 PDF의 대기 청크를 한 핸들 안에서
    # 인용 순위대로 처리합니다 (5 regex 패스는 페이지별 문장 추출에 유지).
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in pending:
        grouped.setdefault(item["file_path"], []).append(item)
    for file_path, items in grouped.items():
        try:
            with open_pdf_document(file_path) as pdf:
                for item in items:
                    page_val = int(item["page_val"])
                    content = str(item["content"])
                    memo_key = item["memo_key"]
                    try:
                        page = pdf[page_val - 1]
                        textpage = page.get_textpage()
                        sentences = _extract_search_sentences(content)

                        doc_quads = []
                        # [최적화] TextPage를 사용하여 고속 검색 (상한: 3문장 x 선두 1프로브)
                        for search_query in sentences:
                            if not search_query:
                                continue

                            logger.info(
                                f"[HIGHLIGHT] Searching query on page {page_val}: '{search_query}'"
                            )

                            part = search_query[:40].strip()
                            if len(part) < 12:
                                continue
                            quads = page.search_for(part, textpage=textpage)
                            if quads:
                                doc_quads.extend(quads)

                        doc_annots: list[dict] = []
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

                                doc_annots.append(
                                    {
                                        "page": page_val,
                                        "x": x_min,
                                        "y": y_min,
                                        "width": x_max - x_min,
                                        "height": y_max - y_min,
                                        "color": "red",
                                    }
                                )
                        annotations.extend(doc_annots)
                        if memo_key is not None:
                            _memo_store(memo_key, doc_annots)
                    except Exception as e:
                        logger.error(f"[HIGHLIGHT] On-demand search failed: {e}")
                        continue
        except Exception as e:
            logger.error(f"[HIGHLIGHT] On-demand search failed: {e}")
            continue

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

    citations: 구조화 인용 배열(doc_id 기반). 본문 말미 블록은 덧붙이지
    않는다 — 답변 말미 출처 표시는 짧은 회색 [N] 배지로 별도 렌더된다.
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
        _sid = doc_stable_id(_doc)
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
            # ACC-6: #0056b3 (7:1 on white) + weight 700 for link contrast.
            f'style="color: #0056b3; font-weight: 700; text-decoration: underline; text-underline-offset: 3px;">'
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
            # ACC-6: #0056b3 (7:1 on white) + weight 700 for link contrast.
            f'style="color: #0056b3; font-weight: 700; text-decoration: underline; text-underline-offset: 3px;">'
            f"{full_match}</span>"
        )

    try:
        # 1. 문서 인용 [doc:N] 치환 (호버 시 툴팁 표시)
        text = _RE_DOC_CITATION.sub(replace_doc_citation, text)
        # 2. 페이지 인용 [p.3] 치환 (호버 시 툴팁 표시)
        text = _RE_CITATION_BLOCK.sub(replace_citation, text)
    except Exception as e:
        logger.error(f"[Utils] 인용구 처리 오류: {e}")

    # 3. 구조화 citations[]는 본문에 덧붙이지 않는다: 답변 말미 출처 표시는
    #    chat_references.render_inline_citation_badges가 짧은 회색 [N] 배지
    #    (발췌는 help= 호버)로 렌더한다. 클릭 불가한 파란 전문 블록 제거.
    #    인라인 [doc:N]/[p.X] 스팬 툴팁(위 단계)은 그대로 유지된다.

    return text


__all__ = [
    "apply_tooltips_to_response",
    "compute_file_hash",
    "count_tokens_rough",
    "doc_stable_id",
    "extract_annotations_from_docs",
    "fast_hash",
    "normalize_latex_delimiters",
    "preprocess_text",
    "run_in_background_worker",
    "safe_cache_data",
    "safe_cache_resource",
    "strip_context_tokens",
    "_build_line_boxes",
]
