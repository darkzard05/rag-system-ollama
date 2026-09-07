"""D19: 결정적 해시 기반 검증 샘플링 (_should_verify) 단위 테스트.

기존 random.random() 샘플링은 실행 간 pass/fail 부분집합이 바뀌어 실패 케이스를
재현할 수 없었다. _should_verify는 query + 정렬된 doc stable-id를 sha256으로
해시해 결정적으로 검증 여부를 판정한다 — 동일 입력은 항상 동일 결정.
"""

from unittest.mock import patch

from core.graph._verify import _should_verify


def test_should_verify_is_deterministic():
    """동일 (query, doc_ids)는 반복 호출해도 항상 동일 결정을 내립니다."""
    query = "문서 인용 검증 질문"
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    decisions = {_should_verify(query, doc_ids) for _ in range(5)}
    assert len(decisions) == 1


def test_should_verify_varies_by_query_or_docs():
    """입력(query/docs)이 다르면 결정이 달라져야 합니다 (입력 민감성).

    rate=0.5로 고정해 brute-force로 찾은 결정적 예제:
    - ("질문 A", ["doc-a", "doc-b"]) → bucket 0.3441 (< 0.5, 검증)
    - ("질문 A", ["doc-a"])          → bucket 0.665  (>= 0.5, 미검증)
    """
    with patch("core.graph._verify.VERIFICATION_SAMPLE_RATE", 0.5):
        assert _should_verify("질문 A", ["doc-a", "doc-b"]) is True
        assert _should_verify("질문 A", ["doc-a"]) is False


def test_sample_rate_zero_never_verifies():
    """VERIFICATION_SAMPLE_RATE=0.0이면 어떤 입력도 검증하지 않습니다."""
    with patch("core.graph._verify.VERIFICATION_SAMPLE_RATE", 0.0):
        for query in ("질문 A", "질문 B", "doc 인용 질문"):
            assert _should_verify(query, ["doc-a"]) is False
        # 빈-문서 쿼리도 결정적 버킷(["",])으로 동일하게 미검증
        assert _should_verify("질문 B", []) is False
