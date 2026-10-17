from langchain_core.documents import Document

from core.graph._retrieval import _merge_adjacent_chunks


def test_merge_adjacent_chunks_preserves_rerank_order():
    """F3: _merge_adjacent_chunks가 리랭커의 관련도 순서(rerank order)를 파괴하지 않고 유지하는지 검증."""
    # 리랭커가 결정한 순서: Doc 2 (page 1, chunk 2)가 Doc 1 (page 1, chunk 1)보다 더 관련성이 높다고 판단함 (rerank_score 순)
    doc_chunk2 = Document(
        page_content="Content of chunk 2",
        metadata={
            "source": "doc.pdf",
            "page": 1,
            "chunk_index": 2,
            "rerank_score": 0.9,
        },
    )
    doc_chunk1 = Document(
        page_content="Content of chunk 1",
        metadata={
            "source": "doc.pdf",
            "page": 1,
            "chunk_index": 1,
            "rerank_score": 0.8,
        },
    )
    # 다른 페이지의 독립된 문서
    doc_other = Document(
        page_content="Content of other page",
        metadata={
            "source": "doc.pdf",
            "page": 2,
            "chunk_index": 1,
            "rerank_score": 0.7,
        },
    )

    # 입력 순서 = 리랭크 순서 (chunk2가 chunk1보다 앞에 옴)
    input_docs = [doc_chunk2, doc_chunk1, doc_other]

    merged = _merge_adjacent_chunks(input_docs)

    # 기대 결과:
    # 1. chunk2와 chunk1은 같은 페이지(1)이고 인접하므로 병합되어야 함.
    # 2. 병합된 결과의 첫 번째 문서(head)는 리랭크 순서 상 먼저 나타난 doc_chunk2(또는 병합된 그룹)여야 함.
    # 기존 코드(sorted 적용)였다면 chunk_index=1 인 chunk1이 앞으로 강제 정렬되었을 것임.
    assert len(merged) == 2
    # 리랭크 순서(chunk2가 앞)가 유지되어야 하므로, 병합된 문서의 본문은 "Content of chunk 2"로 시작해야 함.
    content = merged[0].page_content
    idx2 = content.find("Content of chunk 2")
    idx1 = content.find("Content of chunk 1")
    assert idx2 < idx1, "리랭크 순서(chunk 2가 chunk 1보다 우선)가 파괴되었습니다."
    assert merged[1].metadata["page"] == 2
