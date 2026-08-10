"""
Todo 9 검증 (R4-02): count_tokens_rough 실측 보정 — 한글(비ASCII) 문자당 추정치가
실제 토크나이저(prompt_eval_count) 대비 2.32배 과대였던 것을 1.2~1.5배 수준으로
재보정했는지 검증합니다.

실측 근거: 동일 프롬프트에서 추정 1388 vs 실제 598 (2.32배 과대).
"""

import pytest

from common.utils import count_tokens_rough


@pytest.mark.parametrize(
    "text",
    [
        "한국어 문서에서 토큰 추정이 실제 대비 과도하게 크지 않아야 합니다.",
        "RAG 시스템의 답변 생성 단계에서 컨텍스트 예산을 계산하는 데 사용됩니다.",
        "가나다라마바사아자차카타파하",
    ],
)
def test_count_tokens_rough_korean_within_12_15x_bounds(text):
    """한글(비ASCII) 문자당 추정 토큰이 1.2~1.5 수준(실측 보정 범위)에 머물러야 합니다."""
    est = count_tokens_rough(text)
    non_ascii = sum(1 for c in text if ord(c) > 0x7F)
    assert est >= int(non_ascii * 1.2)
    assert est <= int(non_ascii * 1.5) + 2


def test_count_tokens_rough_below_old_conservative_rate():
    """기존 보수적 가중치(2.5/문자)보다 확실히 낮아 과대추정이 교정되었는지 검증합니다."""
    korean = "한글" * 50  # 100자 순수 한글
    est = count_tokens_rough(korean)
    assert est < len(korean) * 2.0


def test_count_tokens_rough_empty_and_ascii_baseline():
    """빈 텍스트는 0, ASCII 텍스트는 3~4글자당 1토큰 수준을 유지합니다."""
    assert count_tokens_rough("") == 0
    ascii_est = count_tokens_rough("hello world this is a test")
    assert ascii_est <= len("hello world this is a test") / 3.0 + 2
    assert ascii_est >= 1
