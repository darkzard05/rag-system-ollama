"""F6 검증: ``_filter_min_section_len`` 의 50자 임계값 + 내용 인지 보존.

- 50자 이상 문서: 항상 유지된다.
- 50자 미만 저정보 파편(헤더/번호/기호): 드롭된다.
- 50자 미만 수치/단답: 유지된다 (정답 소실 방지).
- 모든 문서가 저정보 파편인 극단 케이스: 최고 점수 1건만 유지된다.
"""

from langchain_core.documents import Document

from core.graph.graph_builder import _MIN_CONTEXT_SECTION_LEN, _filter_min_section_len


def _doc(content: str, score: float = 0.0) -> Document:
    return Document(page_content=content, metadata={"rerank_score": score})


def test_docs_at_or_above_threshold_are_kept() -> None:
    """Given: 50자 이상 문서들 / When: 필터 호출 / Then: 모두 유지."""
    long_a = "A" * _MIN_CONTEXT_SECTION_LEN
    long_b = "B" * (_MIN_CONTEXT_SECTION_LEN + 10)
    docs = [_doc(long_a), _doc(long_b)]

    result = _filter_min_section_len(docs)

    assert len(result) == 2
    assert long_a in {d.page_content for d in result}
    assert long_b in {d.page_content for d in result}


def test_low_info_docs_below_threshold_are_dropped() -> None:
    """Given: 저정보 49자 + 50자 문서 / When: 필터 호출 / Then: 저정보만 드롭."""
    short = "-" * (_MIN_CONTEXT_SECTION_LEN - 1)
    long = "B" * _MIN_CONTEXT_SECTION_LEN
    docs = [_doc(short), _doc(long)]

    result = _filter_min_section_len(docs)

    assert len(result) == 1
    assert result[0].page_content == long


def test_numeric_docs_below_threshold_are_kept() -> None:
    """Given: 수치 단답 49자 이하 / When: 필터 호출 / Then: 정답 단편은 유지."""
    numeric = "671B" + " " * (_MIN_CONTEXT_SECTION_LEN - 10) + "1.76T"
    docs = [_doc(numeric), _doc("B" * _MIN_CONTEXT_SECTION_LEN)]

    result = _filter_min_section_len(docs)

    assert len(result) == 2
    assert numeric in {d.page_content for d in result}


def test_all_low_info_docs_keep_highest_score_one() -> None:
    """Given: 모든 문서가 저정보 파편 / When: 필터 호출 / Then: 최고 점수 1건 유지."""
    docs = [
        _doc("## 1", score=0.1),
        _doc("---", score=0.9),
        _doc("3.", score=0.4),
    ]

    result = _filter_min_section_len(docs)

    assert len(result) == 1
    assert result[0] is docs[1]


def test_empty_input_returns_empty() -> None:
    """Given: 빈 리스트 / When: 필터 호출 / Then: 빈 리스트 반환."""
    assert _filter_min_section_len([]) == []
