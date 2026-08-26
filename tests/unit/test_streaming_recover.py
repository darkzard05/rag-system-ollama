"""
문제 B 회귀: JSON 파싱 실패 시 채팅 버블에 원시 JSON 이 노출되지 않도록
final_answer 를 정규식으로 복구하는 경로를 검증한다.
"""

from __future__ import annotations

from ui.components.streaming import _recover_final_answer


def test_recover_unclosed_final_answer():
    """닫히지 않은 따옴표로 json.loads 실패해도 정답 텍스트를 복구한다."""
    blob = '{"final_answer": "CM3는 조건부 확률 모델이다'  # 닫는 따옴표 없음
    assert _recover_final_answer(blob) == "CM3는 조건부 확률 모델이다"


def test_recover_with_embedded_newline():
    """값 내 줄바꿈이 있어도 복구된다."""
    blob = '{"final_answer": "line1\nline2"'
    assert _recover_final_answer(blob) == "line1\nline2"


def test_recover_closed_value_truncates_at_quote():
    """닫는 따옴표 이후의 잔여 토큰은 버린다."""
    blob = '{"final_answer": "answer text","citations": [...]'
    assert _recover_final_answer(blob) == "answer text"


def test_recover_no_final_answer_returns_none():
    """final_answer 키가 없으면 복구 불가(None)."""
    blob = '{"reasoning": "r","citations": []}'
    assert _recover_final_answer(blob) is None


def test_recover_empty_blob_returns_none():
    assert _recover_final_answer("") is None
