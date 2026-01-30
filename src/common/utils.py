"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
Utils Rebuild: 복잡한 데코레이터 제거 및 비동기 헬퍼 단순화.
"""

import asyncio
import functools
import html
import logging
import re
import time

logger = logging.getLogger(__name__)
# ... (기존 변수 및 함수 유지)


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
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # 2. 인라인 수식 변환: \( ... \) -> $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # 3. 잘못된 이스케이프 문자 정제 (예: \$ -> $)
    # 단, 코드 블록 내의 기호는 건드리지 않도록 주의가 필요하나 일반 답변 기준 처리
    text = text.replace(r"\$", "$")

    return text


def apply_tooltips_to_response(response_text: str, documents: list) -> str:
    """
    LLM 응답 텍스트 내의 인용구([p.X])를 찾아 툴팁 HTML로 변환합니다.
    문서의 내용을 툴팁 텍스트로 삽입합니다.
    동시에 LaTeX 수식 구분자도 정규화합니다.
    """
    if not response_text:
        return response_text

    # 0. 수식 구분자 정규화 (LaTeX 호환성 확보)
    response_text = normalize_latex_delimiters(response_text)

    if not documents:
        return response_text

    # 1. 페이지별 텍스트 매핑 생성
    # 여러 청크가 같은 페이지일 수 있으므로 텍스트를 병합합니다.
    page_content_map = {}
    for doc in documents:
        page = doc.metadata.get("page")
        if not page:
            continue

        # 페이지 번호를 문자열로 통일
        page_key = str(page)
        content = doc.page_content.strip()

        if page_key in page_content_map:
            # 중복 내용은 제외하고 병합
            if content not in page_content_map[page_key]:
                page_content_map[page_key] += "\n\n... " + content
        else:
            page_content_map[page_key] = content

    # 2. 정규표현식으로 인용 패턴 찾기 및 치환
    # 목표: [p.123], (p. 123), [page 123], [p 123] 등을 모두 포착 (대소문자 무관, 공백 허용)
    # Group 1: Opening Bracket
    # Group 2: Page Number
    # Group 3: Closing Bracket
    pattern = re.compile(
        r"([\[\(])(?:Document\s+\d+[,.]?\s*)?(?:[Pp](?:age)?\.?\s*)(\d+)([\]\)])",
        re.IGNORECASE,
    )

    def replacement(match):
        page_num = match.group(2)

        # 표시할 텍스트는 항상 표준화된 형식 [p.X] 으로 통일
        display_text = f"[p.{page_num}]"

        if page_num in page_content_map:
            # HTML Safe 처리
            raw_text = page_content_map[page_num]
            if len(raw_text) > 500:
                raw_text = raw_text[:500] + "..."

            safe_text = html.escape(raw_text).replace("\n", "<br>")

            # [복구] 클릭 기능을 제거하고 툴팁만 제공 (세션 초기화 방지)
            return (
                f'<span class="tooltip">{display_text}'
                f'<span class="tooltip-text">{safe_text}</span>'
                f"</span>"
            )
        else:
            return display_text

    # 3. 괄호가 없는 p.123 형태도 추가로 잡기 위한 2차 패턴 (선택사항, 노이즈 주의)
    # 여기서는 안전하게 괄호가 있는 경우만 먼저 완벽히 처리합니다.
    new_response = pattern.sub(replacement, response_text)

    return new_response


_RE_WHITESPACE = re.compile(r"\s+")
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


def preprocess_text(text: str) -> str:
    """텍스트 정제: 널 문자 및 연속 공백 제거"""
    if not text:
        return ""
    text = text.replace("\x00", "")
    return _RE_WHITESPACE.sub(" ", text).strip()


def clean_query_text(query: str) -> str:
    """쿼리 텍스트에서 불필요한 기호 및 번호 제거"""
    if not query:
        return ""
    # 1. '1.', '2.', '- ', '* ' 등 시작 패턴 제거
    query = re.sub(r"^\d+[\.\)]\s*", "", query)
    query = re.sub(r"^[\-\*•]\s*", "", query)
    # 2. 따옴표 제거
    query = query.replace('"', "").replace("'", "")
    return query.strip()


import streamlit as st


@st.cache_data(ttl=5)  # 5초 동안 리소스 정보 캐싱
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
        return f"🤖 AI 모델 응답 중 오류가 발생했습니다: {msg}"
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
    단순 동기 함수용 로깅 데코레이터.
    GraphBuilder의 Node 함수에는 사용하지 마세요! (config 전달 문제 발생 가능)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[System] [Task] {operation_name} 시작...")
            start = time.time()
            try:
                res = func(*args, **kwargs)
                dur = time.time() - start
                logger.info(f"[System] [Task] {operation_name} 완료 ({dur:.2f}s)")
                return res
            except Exception as e:
                logger.error(f"[System] [Task] {operation_name} 실패: {e}")
                raise

        return wrapper

    return decorator
