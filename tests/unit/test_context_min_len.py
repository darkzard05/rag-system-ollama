"""F6 검증: ``_filter_min_section_len`` 의 50자 임계값 + 단답형 fallback 가드.

- 50자 이상 문서: 유지된다.
- 50자 미만 문서: 드롭된다.
- 모든 문서가 50자 미만인 극단 케이스: 정확히 1개 문서만 유지된다 (가드).
"""

from langchain_core.documents import Document

from core.graph_builder import _MIN_CONTEXT_SECTION_LEN, _filter_min_section_len


def _doc(content: str) -> Document:
    return Document(page_content=content, metadata={})


def test_docs_at_or_above_threshold_are_kept() -> None:
    """Given: 50자 이상 문서들 / When: 필터 호출 / Then: 모두 유지."""
    long_a = "A" * _MIN_CONTEXT_SECTION_LEN
    long_b = "B" * (_MIN_CONTEXT_SECTION_LEN + 10)
    docs = [_doc(long_a), _doc(long_b)]

    result = _filter_min_section_len(docs)

    assert len(result) == 2
    assert long_a in {d.page_content for d in result}
    assert long_b in {d.page_content for d in result}


def test_docs_below_threshold_are_dropped() -> None:
    """Given: 50자 미만 문서 + 50자 이상 문서 / When: 필터 호출 / Then: 짧은 문서만 드롭."""
    short = "A" * (_MIN_CONTEXT_SECTION_LEN - 1)
    long = "B" * _MIN_CONTEXT_SECTION_LEN
    docs = [_doc(short), _doc(long)]

    result = _filter_min_section_len(docs)

    assert len(result) == 1
    assert result[0].page_content == long


def test_all_docs_below_threshold_keeps_exactly_one() -> None:
    """Given: 모든 문서가 50자 미만 / When: 필터 호출 / Then: 정확히 1개 유지 (가드)."""
    short_a = "A" * (_MIN_CONTEXT_SECTION_LEN - 1)
    short_b = "B" * (_MIN_CONTEXT_SECTION_LEN - 20)
    short_c = "C" * 1
    docs = [_doc(short_a), _doc(short_b), _doc(short_c)]

    result = _filter_min_section_len(docs)

    assert len(result) == 1
    assert result[0].page_content in {short_a, short_b, short_c}


def test_empty_input_returns_empty() -> None:
    """Given: 빈 리스트 / When: 필터 호출 / Then: 빈 리스트 반환."""
    assert _filter_min_section_len([]) == []
