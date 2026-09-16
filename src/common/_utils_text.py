"""Domain utilities extracted from ``common.utils`` (Batch 5.4).
Log namespace preserved as ``common.utils`` for compatibility.
"""

import logging
import re

logger = logging.getLogger("common.utils")


_RE_LATEX_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_RE_LATEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
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


# [F5] 포맷 컨텍스트 토큰([doc:..] [section:..] [page:..] [score:..])을 본문에서
# 제거한다. 이 토큰은 검색 컨텍스트용이므로 최종 답변 본문에 그대로 노출되면 지저분.
# 인라인 인용 앵커([1] 등)는 보존한다.
_RE_CONTEXT_TOKEN = re.compile(r"\s*\[(doc|section|page|score):[^\]]*\]")


def strip_context_tokens(text: str) -> str:
    """최종 답변 본문에서 컨텍스트 메타토큰([doc:..] 등)을 제거합니다."""
    if not text:
        return text
    return _RE_CONTEXT_TOKEN.sub(" ", text).strip()


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
