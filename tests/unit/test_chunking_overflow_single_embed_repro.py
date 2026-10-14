"""Repro: 오버플로우 청킹의 임베딩 2회 분할 호출 결함.

결함 경로 (src/core/chunking.py::split_documents, needs_sub_chunking):
  overflow_docs → semantic_chunker.split_documents → embed_documents 1회 (문장)
  normal_docs   → _embed_documents_chunks      → embed_documents 1회 (청크)

Ollama embed_documents는 입력 전체를 단일 HTTP 요청으로 전송하므로
(chunking.py 주석 참조) 분할 호출은 왕복·스레드풀 생성만 늘린다.

계약: 의미분할 활성화 오버플로우 경로에서 embed_documents 호출은 1회이며,
반환 벡터 수는 최종 청크 수와 일치하고 순서가 보존된다.
"""

import hashlib

import numpy as np
import pytest
from langchain_core.documents import Document

import core.chunking as chunking_mod
from core.chunking import split_documents


class CountingEmbedder:
    """embed_documents 호출 횟수를 세는 결정론적 내용-감응 임베더 (dim=8)."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.model = "counting-test-model"
        self.calls: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        vectors = []
        for t in texts:
            digest = hashlib.md5(t.encode()).digest()
            vectors.append([float(b % 5) for b in digest[: self.dim]])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dim


class TrackingCacheManager:
    """디스크 I/O 없는 인메모리 캐시 (청커 주입용)."""

    def __init__(self):
        self._store: dict = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value, persist_to_disk: bool = True) -> None:
        self._store[key] = value


def _make_docs() -> list[Document]:
    docs = [
        Document(
            page_content=f"정상 섹션 {i} 내용입니다. 검색용 짧은 본문.",
            metadata={"is_already_chunked": True, "page": i + 1},
        )
        for i in range(18)
    ]
    topics = ("첫 번째 오버플로우", "두 번째 오버플로우")
    for j in range(2):
        sentences = " ".join(
            f"{topics[j]} 대상 {k}번째 문장 내용은 고유 식별자 {j}-{k}를 포함합니다."
            for k in range(60)
        )
        docs.append(
            Document(
                page_content=sentences,  # ~3000자, 60문장
                metadata={"is_already_chunked": True, "page": 100 + j},
            )
        )
    return docs


@pytest.mark.asyncio
async def test_overflow_path_embeds_once(monkeypatch) -> None:
    """오버플로우+정상 분기에도 embed_documents 호출은 1회여야 한다."""
    embedder = CountingEmbedder()

    real_init = chunking_mod._init_semantic_chunker

    def tracking_init(inner_embedder):
        chunker = real_init(inner_embedder)
        chunker.cache_manager = TrackingCacheManager()
        return chunker

    monkeypatch.setattr(chunking_mod, "_init_semantic_chunker", tracking_init)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": True})
    monkeypatch.setattr(
        chunking_mod,
        "TEXT_SPLITTER_CONFIG",
        {"chunk_size": 500, "chunk_overlap": 50},
    )
    # 배치 분할 효과를 제거해 호출 횟수 = 임베딩 라운드트립 횟수가 되게 한다.
    monkeypatch.setattr(chunking_mod, "_get_optimal_batch_size", lambda e: 1024)
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    split_docs, vectors = await split_documents(_make_docs(), embedder=embedder)

    assert vectors is not None
    assert len(vectors) == len(split_docs)
    assert len(split_docs) > 20  # 오버플로우 2페이지가 하위분할되어야 한다
    total_calls = len(embedder.calls)
    assert total_calls == 1, (
        f"split-embed defect: embed_documents가 {total_calls}회 호출됨 "
        f"(크기 목록: {embedder.calls}, 기대: 1회)"
    )
    for v in vectors:
        assert np.asarray(v).shape[0] == 8


@pytest.mark.asyncio
async def test_overflow_splits_match_direct_semantic_path(monkeypatch) -> None:
    """단일-호출 경로의 오버플로우 분할 결과가 직접 의미분할과 동일해야 한다."""
    from core.semantic_chunker import EmbeddingBasedSemanticChunker

    embedder = CountingEmbedder()
    docs = _make_docs()
    overflow_docs = [d for d in docs if len(d.page_content) > 750]
    normal_docs = [d for d in docs if len(d.page_content) <= 750]
    assert len(overflow_docs) == 2
    assert len(normal_docs) == 18

    real_init = chunking_mod._init_semantic_chunker

    def tracking_init(inner_embedder):
        chunker = real_init(inner_embedder)
        chunker.cache_manager = TrackingCacheManager()
        return chunker

    monkeypatch.setattr(chunking_mod, "_init_semantic_chunker", tracking_init)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": True})
    monkeypatch.setattr(
        chunking_mod,
        "TEXT_SPLITTER_CONFIG",
        {"chunk_size": 500, "chunk_overlap": 50},
    )
    monkeypatch.setattr(chunking_mod, "_get_optimal_batch_size", lambda e: 1024)
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    split_docs, vectors = await split_documents(docs, embedder=embedder)

    # 직접 의미분할(구 경로와 동일한 구조 계산)과 오버플로우 청크 집합이 일치.
    direct = EmbeddingBasedSemanticChunker(embedder=embedder, batch_size=1024)
    direct.cache_manager = TrackingCacheManager()
    expected_splits, _ = await direct.split_documents(overflow_docs)
    got_overflow = sorted(
        d.page_content
        for d in split_docs
        if d.page_content not in {n.page_content for n in normal_docs}
    )
    want_overflow = sorted(d.page_content for d in expected_splits)
    assert got_overflow == want_overflow

    # 정상 문서는 원본 순서대로 앞에 보존된다.
    assert [d.page_content for d in split_docs[:18]] == [
        d.page_content for d in normal_docs
    ]
    assert len(vectors) == len(split_docs)
