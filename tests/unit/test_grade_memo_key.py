"""Top3-Refactor M1: GRADE_MEMO_KEY 단일 정의 계약 검증 테스트."""

from common.constants import GRADE_MEMO_KEY as CONSTANTS_GRADE_MEMO_KEY
from core.graph import _grade
from core.pipeline_builder import GRADE_MEMO_KEY as PIPELINE_GRADE_MEMO_KEY


def test_grade_memo_key_single_source_of_truth() -> None:
    """모든 소비처가 common.constants 단일 정의를 참조해야 한다."""
    assert CONSTANTS_GRADE_MEMO_KEY == "grade_decision_memo"
    assert PIPELINE_GRADE_MEMO_KEY == CONSTANTS_GRADE_MEMO_KEY
    assert _grade._GRADE_MEMO_KEY == CONSTANTS_GRADE_MEMO_KEY


def test_grade_memo_key_consumers_match_value() -> None:
    """소비처 키 값이 단일 정의와 동일해야 한다 (리터럴 재정의 시 파손)."""
    assert PIPELINE_GRADE_MEMO_KEY == "grade_decision_memo"
    assert _grade._GRADE_MEMO_KEY == "grade_decision_memo"
