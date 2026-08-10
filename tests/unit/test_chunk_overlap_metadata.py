"""R2-05 검증: 오버랩 청크의 start/end/pages가 실제 텍스트 범위와 정렬되는지 검증.

버그: ``split_text``가 청크 텍스트(:812-814, actual_start 기준)에 오버랩 문장을
포함하면서도 ``start`` 오프셋(:824)을 overlap 제외(group_start 기준)로 산출한다.
그 결과 오버랩 문장 구간에 대한 좌표·페이지 메타데이터가 실제 텍스트 범위보다
좁아져, 페이지 경계 오버랩 시 해당 문장의 하이라이트 좌표가 누락될 수 있다.
"""

import numpy as np
import pytest
from langchain_core.documents import Document

from core.semantic_chunker import EmbeddingBasedSemanticChunker

# 문장 길이 30자 이상 (MIN_MERGE_LEN 병합 회피) + 3A+3B 구조에서 결정적 1개 분기점 유도
A1 = "Alpha first sentence within group a for the overlap test."
A2 = "Alpha second sentence within group a for the overlap test."
A3 = "Alpha third sentence within group a for the overlap test."
B1 = "Beta first sentence within group b for the overlap test."
B2 = "Beta second sentence within group b for the overlap test."
B3 = "Beta third sentence within group b for the overlap test."


class OrthoEmbeddings:
    """group-a/group-b 문장을 직교 벡터로 임베딩해 결정적 분할을 유도합니다."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        self.model_name = "ortho-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = np.zeros(self.dimension)
            lowered = text.lower()
            if "group a" in lowered:
                vec[0] = 1.0
            elif "group b" in lowered:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            results.append(vec.tolist())
        return results


def _make_chunker(chunk_overlap: int) -> EmbeddingBasedSemanticChunker:
    # buffer_size=0: 결합 벡터가 문장 벡터 그대로여서 A→B 경계 거리가 1.0으로
    # 결정적입니다 (buffer>0이면 창 평균 정규화로 경계 거리가 0.2까지 희석됨).
    return EmbeddingBasedSemanticChunker(
        embedder=OrthoEmbeddings(),
        chunk_overlap=chunk_overlap,
        buffer_size=0,
        min_chunk_size=0,
        max_chunk_size=5000,
        breakpoint_threshold_type="similarity_threshold",
        similarity_threshold=0.5,
    )


@pytest.mark.asyncio
async def test_overlap_chunk_offsets_match_actual_text_range():
    """오버랩 문장을 포함하는 청크의 start/end는 실제 텍스트 범위와 일치해야 합니다.

    overlap=1일 때 청크2는 [A3, B1, B2, B3]이므로 start는 A3의 시작 오프셋,
    end는 B3의 끝 오프셋이어야 합니다 (기존: A3 미포함 범위).
    """
    chunker = _make_chunker(chunk_overlap=1)
    text = " ".join([A1, A2, A3, B1, B2, B3])
    chunks = await chunker.split_text(text)

    assert len(chunks) == 2, f"3A+3B 구조에서 2개 청크가 되어야 합니다: {chunks!r}"
    overlap_chunk = chunks[1]
    # 오버랩 문장 A3가 실제로 청크 텍스트에 포함되어 있는지 확인 (전제 조건)
    assert overlap_chunk["text"].startswith(A3)

    assert overlap_chunk["start"] == text.index(A3), (
        f"오버랩 포함 청크의 start가 실제 텍스트 범위(A3 시작)와 일치해야 합니다. "
        f"기대: {text.index(A3)}, 실제: {overlap_chunk['start']}"
    )
    assert overlap_chunk["end"] == text.index(B3) + len(B3), (
        f"오버랩 포함 청크의 end는 마지막 문장(B3) 끝과 일치해야 합니다. "
        f"기대: {text.index(B3) + len(B3)}, 실제: {overlap_chunk['end']}"
    )


@pytest.mark.asyncio
async def test_overlap_chunk_pages_include_overlap_sentence_page():
    """페이지 경계 오버랩: 오버랩 문장이 이전 페이지에 있으면 pages에 포함되어야 합니다.

    페이지1(A1, A2) / 페이지2(B1, B2, B3) 구성에서 분기점이 페이지 경계에 생기고,
    overlap=1이면 청크2 = [A2, B1, B2, B3]가 되어 pages=[1, 2]여야 합니다.
    (기존: start가 B1 기준이라 pages=[2] — A2 좌표 하이드레이션 누락)
    """
    chunker = _make_chunker(chunk_overlap=1)
    docs = [
        Document(page_content=f"{A1} {A2}", metadata={"page": 1}),
        Document(page_content=f"{B1} {B2} {B3}", metadata={"page": 2}),
    ]
    final_docs, _ = await chunker.split_documents(docs)

    assert len(final_docs) == 2
    overlap_chunk = final_docs[1]
    assert overlap_chunk.page_content.startswith(A2), (
        f"오버랩 문장(A2)이 청크2 텍스트에 포함되어야 합니다: "
        f"{overlap_chunk.page_content[:60]!r}"
    )
    assert overlap_chunk.metadata["pages"] == [1, 2], (
        f"페이지 경계 오버랩 시 pages가 두 페이지를 모두 포함해야 합니다 "
        f"(실제: {overlap_chunk.metadata.get('pages')})"
    )


@pytest.mark.asyncio
async def test_no_overlap_chunk_pages_remain_without_overlap_page():
    """대조군: overlap=0이면 청크2=[B1, B2, B3]이며 pages=[2]여야 합니다 (회귀 방지)."""
    chunker = _make_chunker(chunk_overlap=0)
    docs = [
        Document(page_content=f"{A1} {A2}", metadata={"page": 1}),
        Document(page_content=f"{B1} {B2} {B3}", metadata={"page": 2}),
    ]
    final_docs, _ = await chunker.split_documents(docs)

    assert len(final_docs) == 2
    chunk2 = final_docs[1]
    assert chunk2.page_content.startswith(B1)
    assert chunk2.metadata["pages"] == [2]
    # 오버랩이 없는 청크는 start/end_index가 B 구간과 일치
    assert chunk2.metadata["start_index"] == chunk2.metadata["start_index"]
