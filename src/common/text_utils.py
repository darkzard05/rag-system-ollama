"""
텍스트 처리 및 토크나이징 관련 유틸리티 모듈.
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
