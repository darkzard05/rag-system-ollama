"""
텍스트 처리 및 토크나이징 관련 유틸리티 모듈.

일반 텍스트 유틸(LaTeX 정규화, 컨텍스트 토큰 제거, 전처리, 토큰 추정)은
본 모듈이 단일 진원이다. ``common.utils`` 는 하위 호환을 위해 재수출한다.
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- 최적화된 토크나이저 ---
_RE_KOREAN_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+")


def bm25_tokenizer(text: str) -> list[str]:
    """
    [최적화] 한국어 검색 품질 향상을 위한 Hybrid 토크나이저.
    기본 정규식 추출 + 어미 제거 + Bi-gram 생성을 수행합니다.
    """
    if not text:
        return []

    # 1. 기본 토큰 추출
    tokens = _RE_KOREAN_TOKEN.findall(text.lower())
    if not tokens:
        return text.split()

    final_tokens = []
    # 자주 쓰이는 조사/어미 (간이 불용어 처리)
    particles = (
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "로",
        "서",
        "들",
        "께서",
        "에서",
        "보다",
        "부터",
        "까지",
        "에게",
        "한테",
    )

    for token in tokens:
        final_tokens.append(token)

        # 한글인 경우 추가 처리
        if "가" <= token[0] <= "힣":
            # 2. 간단한 어미/조사 제거 (끝글자 체크)
            # 2음절 이상의 조사 처리 지원
            for p_len in [2, 1]:
                if len(token) > p_len + 1 and token.endswith(particles):
                    # particles 튜플에 2음절 조사가 포함되어 있으므로 endswith가 올바르게 작동함
                    stem = token[:-p_len]
                    if len(stem) >= 2:
                        final_tokens.append(stem)
                        break

            # 3. Bi-gram 생성 (3글자 이상인 경우)
            # 복합명사 검색 재현율 향상 (예: 인공지능 -> 인공, 공지, 지능)
            if len(token) >= 3:
                for i in range(len(token) - 1):
                    final_tokens.append(token[i : i + 2])

    # 중복 제거 및 짧은 토큰 필터링 (불용어 제외)
    return list(dict.fromkeys(final_tokens))


def has_bm25_tokens(corpus: list[str]) -> bool:
    """코퍼스에 BM25 토크나이저가 생성할 유효 토큰이 하나라도 있는지 확인합니다.

    ``rank_bm25`` 는 모든 문서가 빈 토큰 목록을 가질 때 ``nd``(단어→문서빈도)가
    비어 ``self.idf`` 가 ``{}`` 가 되고, ``BM25Okapi._calc_idf`` 에서
    ``idf_sum / len(self.idf)`` 가 0으로 나뉘어 ``ZeroDivisionError`` 를 일으킵니다.
    이 조건을 사전에 감지해 BM25 생성을 우회(벡터 전용 폴백)하는 데 사용합니다.
    """
    return any(bm25_tokenizer(text) for text in corpus)


# --- 일반 텍스트 유틸 (Phase 1A에서 본 모듈로 병합) ---

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
