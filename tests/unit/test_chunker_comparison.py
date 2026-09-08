"""Header-aware chunking comparison — migrated from scripts/compare_chunking_logic.py.

원본 스크립트는 ``FakeEmbeddings`` + ``EmbeddingBasedSemanticChunker`` 를 사용해
헤더 인식 청킹 결과를 print 로만 확인했습니다. 여기서는 완전 결정적(해시 기반)
목 임베더를 사용해 헤더 인식성(header-awareness)과 ``current_section``
메타데이터를 실제 ``assert`` 로 검증합니다. 라이브 Ollama/네트워크 의존성 없음.
"""

import hashlib

import numpy as np
import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class DeterministicMockEmbeddings:
    """텍스트 해시로 시드된 고정 벡터를 반환하는 결정적 목 임베더."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.model_name = "mock-hash-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(
                hashlib.md5(text.lower().encode()).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(self.dimension).tolist())
        return vectors


SAMPLE_TEXT = """
# Introduction
This is the first sentence of the introduction. Deep learning has revolutionized many fields.
The second sentence explains how it works.

# Methodology
We propose a new method for RAG systems. This involves several steps.
First, we analyze the structure. Second, we apply semantic chunking.

## Preprocessing
Data cleaning is the first step. We remove noise and outliers.
The results show improved accuracy.
""".strip()


def _build_chunker() -> EmbeddingBasedSemanticChunker:
    return EmbeddingBasedSemanticChunker(
        embedder=DeterministicMockEmbeddings(),
        max_chunk_size=500,
        similarity_threshold=0.5,
    )


@pytest.mark.asyncio
async def test_header_aware_chunking_sets_current_section_metadata() -> None:
    chunks = await _build_chunker().split_text(SAMPLE_TEXT)

    assert chunks, "semantic chunker should produce at least one chunk"

    # 모든 청크가 current_section 메타데이터를 가져야 한다.
    for chunk in chunks:
        assert isinstance(chunk.get("current_section"), str)
        assert chunk["current_section"], "current_section must not be empty"
        assert chunk.get("text", "").strip()

    # 섹션 추적: 실제 섹션 제목이 청크 메타데이터에 반영되어야 한다.
    sections = [str(c["current_section"]).upper() for c in chunks]
    assert any("INTRODUCT" in s for s in sections), f"missing Introduction: {sections}"
    assert any("METHOD" in s for s in sections), f"missing Methodology: {sections}"
    assert any("PREPROCESS" in s for s in sections), (
        f"missing Preprocessing: {sections}"
    )


@pytest.mark.asyncio
async def test_headers_never_appear_mid_chunk() -> None:
    chunks = await _build_chunker().split_text(SAMPLE_TEXT)

    assert chunks

    header_starting_chunks = 0
    for chunk in chunks:
        text = chunk["text"].strip()
        # 공백+`#` 패턴(" #")은 헤더가 청크 중간에 끼어 있음을 의미한다.
        assert " #" not in text, f"markdown header found mid-chunk: {text[:80]!r}"
        if text.startswith("#"):
            header_starting_chunks += 1

    # 헤더가 강제 분할 지점으로 작동해 최소 1개 청크는 헤더로 시작해야 한다.
    assert header_starting_chunks > 0
