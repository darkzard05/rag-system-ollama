"""용어 조회(term lookup) 판정 회귀 테스트.

하이픈이 포함된 기술 용어(``vqvae-gan`` 등)가 키워드 패턴(숫자/버전/hex)에
걸리지 않아 확장 재생성을 유발하던 문제를 고정한다.
"""

import pytest

from common.query_signals import is_term_lookup_query


@pytest.mark.parametrize(
    "query",
    [
        "cm3",
        "resnet50",
        "gpt-4o",
        "vqvae-gan",
        "vq-vae",
        "VQVAE-GAN",
        "bert-base",
        "nomic-embed-text-v2-moe",
    ],
)
def test_term_lookup_true(query: str) -> None:
    assert is_term_lookup_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "attention",
        "state-of-the-art",
        "what is vqvae-gan",
        "vqvae-gan이 뭔가요",
        "vqvae-gan 설명해줘",
        "a-b",
        "x",
        "",
        "   ",
    ],
)
def test_term_lookup_false(query: str) -> None:
    assert is_term_lookup_query(query) is False
