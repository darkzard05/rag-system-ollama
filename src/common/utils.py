"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
Utils Rebuild: 복잡한 데코레이터 제거 및 비동기 헬퍼 단순화.
"""

import asyncio
import functools
import hashlib
import logging
import os
import re
import time

import streamlit as st

logger = logging.getLogger(__name__)

# --- 사전 컴파일된 정규표현식 (성능 최적화) ---
_RE_LATEX_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_RE_LATEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

# [수정] 복합 인용 패턴 지원: [p.1, p.2] 또는 [p.1, 2, 3] 또는 [page 5] 등 지원
_RE_CITATION_BLOCK = re.compile(
    r"([\[\(])((?:[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*)(?:(?:[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*))*)([\]\)])",
    re.IGNORECASE,
)
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


def normalize_latex_delimiters(text: str) -> str:
    r"""
    LLM이 출력하는 다양한 LaTeX 수식 구분자를 Streamlit 표준($ 또는 $$)으로 변환합니다.
    - \( ... \) -> $ ... $ (인라인)
    - \[ ... \] -> $$ ... $$ (블록)
    - 기호 앞뒤의 불필요한 이스케이프 제거
    """
    if not text:
        return text

    # 1. 블록 수식 변환: \[ ... \] -> $$ ... $$
    text = _RE_LATEX_BLOCK.sub(r"$$\1$$", text)

    # 2. 인라인 수식 변환: \( ... \) -> $ ... $
    text = _RE_LATEX_INLINE.sub(r"$\1$", text)

    # 3. 잘못된 이스케이프 문자 정제 (예: \$ -> $)
    # 단, 코드 블록 내의 기호는 건드리지 않도록 주의가 필요하나 일반 답변 기준 처리
    text = text.replace(r"\$", "$")

    return text


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
        page_val = int(meta.get("page", 1))
        all_coords = meta.get("word_coords", [])
        content = doc.page_content.lower()

        if not all_coords:
            continue

        # [고도화] 연속성 기반 텍스트 매칭 (Sequence Matching)
        # 1. 청크 텍스트와 PDF 텍스트를 순수 단어 토큰으로 정규화
        content_tokens = re.findall(r"[\w\d]+", content)
        if not content_tokens:
            continue

        pdf_tokens = [re.sub(r"[^\w\d]", "", str(c[4]).lower()) for c in all_coords]

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
            filtered_coords = all_coords[best_start : best_start + len(content_tokens)]
        else:
            # 매칭 실패 시에만 기존의 루즈한 필터링으로 폴백 (최소 가시성)
            filtered_coords = [
                c
                for c in all_coords
                if re.sub(r"[^\w\d]", "", str(c[4]).lower()) in content_tokens[:50]
            ]

        if not filtered_coords:
            continue

        # 4. 줄 단위 그룹화 및 박스 생성
        lines: dict[int, list] = {}
        for c in filtered_coords:
            y_key = round(c[1] / 8) * 8
            if y_key not in lines:
                lines[y_key] = []
            lines[y_key].append(c)

        doc_anno_count = 0
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
                        "thickness": 2,
                    }
                )
                doc_anno_count += 1

        logger.info(
            f"[HIGHLIGHT] Page {page_val}: Found chunk sequence at index {best_start}, created {doc_anno_count} line boxes"
        )

    return annotations


def apply_tooltips_to_response(
    response_text: str, documents: list | None = None, msg_index: int = 0
) -> str:
    """
    답변 내의 인용구([1], [p.5] 등)를 찾아 문서 정보 툴팁을 입힙니다.
    """
    if not response_text:
        return response_text

    # 1. LaTeX 정규화 먼저 수행
    text = normalize_latex_delimiters(response_text)

    # 2. 문서 정보가 없으면 정규화된 텍스트만 반환
    if not documents:
        return text

    def replace_citation(match):
        full_match = match.group(0)
        inner_text = match.group(2)

        # 페이지 번호 추출
        page_matches = _RE_EXTRACT_PAGES.findall(inner_text)
        if not page_matches:
            return full_match

        target_page = int(page_matches[0])

        # 해당 페이지와 일치하는 문서 청크 찾기
        matching_doc = None
        for doc in documents:
            meta = (
                getattr(doc, "metadata", {})
                if hasattr(doc, "metadata")
                else doc.get("metadata", {})
            )
            doc_page = int(meta.get("page", -1))
            if doc_page == target_page:  # [수정] 둘 다 1-indexed이므로 직접 비교
                matching_doc = doc
                break

        if matching_doc:
            content = (
                getattr(matching_doc, "page_content", "")
                if hasattr(matching_doc, "page_content")
                else matching_doc.get("page_content", "")
            )
            # 툴팁용 텍스트 정제 (HTML 태그 및 따옴표 처리)
            clean_content = (
                content.replace('"', "&quot;").replace("'", "&apos;")[:200] + "..."
            )

            # 하이라이트 스타일 적용된 HTML 반환
            return f'<span class="citation-tooltip" title="{clean_content}" style="color: #1e88e5; font-weight: bold; cursor: help; border-bottom: 1px dotted #1e88e5;">{full_match}</span>'

        return full_match

    # 인용구 패턴 매칭 및 치환
    try:
        text = _RE_CITATION_BLOCK.sub(replace_citation, text)
    except Exception as e:
        logger.error(f"[Utils] 인용구 처리 오류: {e}")

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


def clean_query_text(query: str) -> str:
    """쿼리 텍스트에서 불필요한 기호, 번호, 접두사(Example:, Question: 등) 제거"""
    if not query:
        return ""

    # 1. 문두의 숫자, 불렛, 접두사(Example:, Query: 등) 일괄 제거
    query = _RE_QUERY_CLEAN_PREFIX.sub("", query.strip())

    # 2. 문두/문미 따옴표 제거
    query = _RE_QUERY_CLEAN_QUOTES.sub("", query.strip())

    return query.strip()


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


@safe_cache_data(ttl=5)  # 5초 동안 리소스 정보 캐싱
def get_ollama_resource_usage(model_name: str) -> str:
    """
    Ollama API를 통해 특정 모델의 리소스 사용 상태(GPU/CPU)를 조회합니다.
    """
    try:
        import requests

        from common.config import OLLAMA_BASE_URL

        # Ollama ps API 호출
        response = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            for m in models:
                if model_name in m.get("name", ""):
                    size_vram = m.get("size_vram", 0)
                    size = m.get("size", 1)

                    # VRAM 사용 비율 계산
                    vram_ratio = (size_vram / size) * 100
                    if vram_ratio >= 90:
                        return f"GPU (VRAM {vram_ratio:.1f}%)"
                    elif vram_ratio > 0:
                        return f"Hybrid (VRAM {vram_ratio:.1f}%, CPU {100 - vram_ratio:.1f}%)"
                    else:
                        return "CPU (0% VRAM)"

            return "Unknown (Not running)"
        return "Unknown (API Error)"
    except Exception:
        return "Unknown (Connection Error)"


def format_error_message(e: Exception) -> str:
    """
    발생한 예외 객체를 분석하여 사용자에게 보여줄 친절한 메시지를 반환합니다.
    """
    from common.exceptions import (
        EmbeddingModelError,
        EmptyPDFError,
        InsufficientChunksError,
        LLMInferenceError,
    )

    err_type = type(e).__name__
    msg = str(e)

    # 1. 커스텀 도메인 예외 처리
    if isinstance(e, EmptyPDFError):
        return "📄 PDF 파일에 텍스트가 없거나 이미지로만 구성되어 있습니다. 다른 파일을 시도해 보세요."
    elif isinstance(e, InsufficientChunksError):
        return "⚠️ 문서의 유효한 텍스트가 너무 적어 분석할 수 없습니다."
    elif isinstance(e, LLMInferenceError):
        return f"🤖 추론 모델 응답 중 오류가 발생했습니다: {msg}"
    elif isinstance(e, EmbeddingModelError):
        return "🧠 임베딩 모델 로드에 실패했습니다. 자원(VRAM/RAM)이 부족한지 확인해 주세요."

    # 2. 일반 시스템 예외 처리
    if "ConnectionError" in err_type or "11434" in msg:
        return (
            "🔌 Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해 주세요."
        )
    elif "timeout" in msg.lower():
        return (
            "⌛ 처리 시간이 너무 오래 걸려 중단되었습니다. 잠시 후 다시 시도해 주세요."
        )
    elif "out of memory" in msg.lower() or "CUDA" in msg:
        return "🚀 GPU 메모리(VRAM)가 부족합니다. 다른 프로그램을 종료하거나 모델을 작은 것으로 바꿔보세요."

    # 3. 기본값
    return f"❌ 알 수 없는 오류 발생 ({err_type}): {msg}"


def fast_hash(text: str, length: int = 16) -> str:
    """
    보안이 필요 없는 단순 식별용 고속 해시 함수.
    SHA256보다 훨씬 빠른 MD5를 사용하고 결과 길이를 조절합니다.
    """
    if not text:
        return "0" * length
    # usedforsecurity=False: 보안 진단 도구(Bandit 등)에 이 해시가
    # 암호화나 보안 목적으로 사용되지 않음을 알립니다.
    return hashlib.md5(text.encode(errors="ignore"), usedforsecurity=False).hexdigest()[
        :length
    ]


def count_tokens_rough(text: str) -> int:
    """
    텍스트의 토큰 수를 대략적으로 계산합니다.
    - 영어: 약 4글자당 1토큰
    - 한글/특수문자: 약 1~2글자당 1토큰
    보수적으로 계산하기 위해 (글자 수 / 2.5)를 사용합니다.
    """
    if not text:
        return 0
    return int(len(text) / 2.5) + 1


@safe_cache_data(ttl=4)
def _get_cached_pdf_bytes(pdf_path: str) -> bytes | None:
    """PDF 파일 내용을 메모리에 캐싱합니다. (I/O 절감)"""
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            return f.read()
    return None


def sync_run(coro):
    """
    Streamlit(동기 환경)에서 비동기 코루틴을 안전하게 실행하기 위한 헬퍼.
    전역적으로 nest_asyncio가 적용되어 있어야 작동합니다.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


def log_operation(operation_name):
    """
    동기 및 비동기 함수를 모두 지원하는 로깅 데코레이터.
    GraphBuilder의 Node 함수에는 사용하지 마세요! (config 전달 문제 발생 가능)
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"[SYSTEM] [TASK] {operation_name} 시작")
                start = time.time()
                try:
                    res = await func(*args, **kwargs)
                    dur = time.time() - start
                    logger.info(
                        f"[SYSTEM] [TASK] {operation_name} 완료 | 소요: {dur:.2f}s"
                    )
                    return res
                except Exception as e:
                    logger.info(f"[SYSTEM] [TASK] {operation_name} 실패 | {e}")
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger.info(f"[SYSTEM] [TASK] {operation_name} 시작")
                start = time.time()
                try:
                    res = func(*args, **kwargs)
                    dur = time.time() - start
                    logger.info(
                        f"[SYSTEM] [TASK] {operation_name} 완료 | 소요: {dur:.2f}s"
                    )
                    return res
                except Exception as e:
                    logger.info(f"[SYSTEM] [TASK] {operation_name} 실패 | {e}")
                    raise

            return sync_wrapper

    return decorator
