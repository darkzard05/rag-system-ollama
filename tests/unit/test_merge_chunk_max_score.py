"""병합 문서 rerank_score 결정 규칙 유닛 테스트 (R3b-05).

- 병합 문서의 `rerank_score`는 head 청크 점수가 아닌 그룹 내 **최대값**으로 결정된다.
- 서로 다른 섹션 그룹은 각자 자신의 그룹 최대값을 보존한다.
"""

from langchain_core.documents import Document

from core.graph_builder import _merge_adjacent_chunks


def _doc(content: str, chunk_index: int, section: str, score: float) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": "s1",
            "page": 1,
            "chunk_index": chunk_index,
            "current_section": section,
            "rerank_score": score,
        },
    )


def test_merged_chunk_rerank_score_takes_group_max():
    """같은 섹션의 연속 청크가 병합되면 rerank_score는 그룹 최대값이 된다."""
    docs = [
        _doc("A", 0, "sec", 0.3),  # head (낮은 점수)
        _doc("B", 1, "sec", 0.9),  # 그룹 최고점
        _doc("C", 2, "sec", 0.5),
    ]

    merged = _merge_adjacent_chunks(docs, max_tokens=1000)

    assert len(merged) == 1
    assert merged[0].metadata["rerank_score"] == 0.9


def test_separate_groups_keep_own_max_scores():
    """서로 다른 섹션 그룹은 각자 그룹 내 최대 rerank_score를 유지한다."""
    docs = [
        _doc("A", 0, "sec1", 0.6),
        _doc("B", 1, "sec1", 0.2),  # 그룹1 최고는 head(0.6) — head 유지와 동일
        _doc("C", 2, "sec2", 0.4),
        _doc("D", 3, "sec2", 0.8),  # 그룹2 최고는 tail(0.8) — head 승계면 0.4로 오염
    ]

    merged = _merge_adjacent_chunks(docs, max_tokens=1000)

    assert len(merged) == 2
    assert merged[0].metadata["rerank_score"] == 0.6
    assert merged[1].metadata["rerank_score"] == 0.8


def test_single_chunk_keeps_own_score():
    """병합이 일어나지 않는 단일 문서는 자기 rerank_score를 유지한다 (그룹 max = 자기 자신)."""
    docs = [_doc("A", 0, "sec", 0.42)]

    merged = _merge_adjacent_chunks(docs, max_tokens=1000)

    assert len(merged) == 1
    assert merged[0].metadata["rerank_score"] == 0.42
