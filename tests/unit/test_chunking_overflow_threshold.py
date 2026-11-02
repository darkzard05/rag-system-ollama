"""페이지 오버플로우 임계치 회귀 테스트 (토큰 기준).

문자 수(750/2000) 기준은 한국어에서 실제 임베딩 토큰을 과소평가해
v2-moe 512 컨텍스트를 초과하는 청크를 만들었다. 이 테스트는 페이지
하위분할 판정과 최종 청크 토큰 상한이 모두 토큰 기준으로 동작함을 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import core.chunking as chunking_mod  # noqa: E402
from core.chunking import split_documents  # noqa: E402

TOKEN_LIMIT = 448


def _page(index: int, chars: int) -> Document:
    return Document(
        page_content="단락 내용입니다. " * (chars // 10),
        metadata={"is_already_chunked": True, "page": index + 1},
    )


class _FakeSplitter:
    """RecursiveCharacterTextSplitter 대역 — 고정 조각을 반환한다."""

    def __init__(self, parts: list[Document]) -> None:
        self.parts = parts

    def split_documents(self, docs: list[Document]) -> list[Document]:
        return self.parts


def _patch_splitter(monkeypatch: pytest.MonkeyPatch, parts: list[Document]) -> None:
    def _factory(*args: object, **kwargs: object) -> _FakeSplitter:
        return _FakeSplitter(parts)

    monkeypatch.setattr(chunking_mod, "_recursive_splitter_cls", lambda: _factory)


def _patch_config(monkeypatch: pytest.MonkeyPatch, limit: int = TOKEN_LIMIT) -> None:
    # 실제 HF 토크나이저 로드 없이 임계치 의미만 검증한다.
    monkeypatch.setattr(
        chunking_mod, "_embedding_token_count", chunking_mod.count_tokens_rough
    )
    monkeypatch.setattr(
        chunking_mod,
        "_embedding_token_counts",
        lambda texts: [chunking_mod.count_tokens_rough(t) for t in texts],
    )
    monkeypatch.setattr(
        chunking_mod,
        "TEXT_SPLITTER_CONFIG",
        {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "chunk_size_tokens": 512,
            "embedding_token_limit": limit,
        },
    )


@pytest.mark.asyncio
async def test_pages_under_token_limit_are_not_sub_chunked(monkeypatch):
    """토큰 상한 이하 페이지는 원본 페이지 구조를 그대로 유지한다."""
    docs = [_page(i, chars=200) for i in range(5)]
    _patch_config(monkeypatch)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": False})
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    splitter_cls = MagicMock()
    monkeypatch.setattr(chunking_mod, "_recursive_splitter_cls", splitter_cls)

    split_docs, vectors = await split_documents(docs, embedder=None)

    assert len(split_docs) == len(docs)
    assert vectors is None
    splitter_cls.assert_not_called()


@pytest.mark.asyncio
async def test_pages_over_token_limit_are_sub_chunked(monkeypatch):
    """문자 수가 아니라 토큰 상한을 넘는 페이지는 하위 분할한다."""
    docs = [_page(0, chars=1200)]
    _patch_config(monkeypatch)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": False})
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    parts = [
        Document(
            page_content="짧은 조각입니다. " * 5,
            metadata={"page": 1, "is_already_chunked": True},
        )
        for _ in range(3)
    ]
    _patch_splitter(monkeypatch, parts)

    split_docs, _ = await split_documents(docs, embedder=None)

    assert len(split_docs) == 3


@pytest.mark.asyncio
async def test_char_based_750_boundary_is_no_longer_used(monkeypatch):
    """750자 초과 여부가 아니라 토큰 상한 초과 여부로 판정한다."""
    docs = [_page(0, chars=750)]
    _patch_config(monkeypatch, limit=TOKEN_LIMIT)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": False})
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    parts = [
        Document(
            page_content="짧은 조각입니다. " * 5,
            metadata={"page": 1, "is_already_chunked": True},
        )
        for _ in range(2)
    ]
    _patch_splitter(monkeypatch, parts)

    split_docs, _ = await split_documents(docs, embedder=None)

    assert len(split_docs) == 2


@pytest.mark.asyncio
async def test_enforce_token_limit_reembeds_split_parts_in_one_batch(monkeypatch):
    """최종 상한 강제는 초과 문서만 재분할하고, 새 조각은 한 번에 재임베딩한다."""
    _patch_config(monkeypatch)
    short_a = Document(page_content="짧은 문서 A", metadata={"page": 1})
    long_doc = Document(page_content="가" * 400, metadata={"page": 2})
    short_b = Document(page_content="짧은 문서 B", metadata={"page": 3})
    vectors = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([2.0, 2.0]),
    ]
    parts = [
        Document(page_content="조각 1", metadata={"page": 2}),
        Document(page_content="조각 2", metadata={"page": 2}),
    ]
    _patch_splitter(monkeypatch, parts)

    batches: list[list[Document]] = []

    async def fake_embed(chunks: list[Document], embedder: object):
        batches.append(list(chunks))
        return [np.array([9.0, 9.0]) for _ in chunks]

    monkeypatch.setattr(chunking_mod, "_embed_documents_chunks", fake_embed)

    out_docs, out_vectors = await chunking_mod._enforce_embedding_token_limit(
        [short_a, long_doc, short_b], vectors, embedder=object()
    )

    assert len(out_docs) == 4
    assert [d.metadata.get("split_from_oversized") for d in out_docs] == [
        None,
        True,
        True,
        None,
    ]
    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert out_vectors is not None
    assert np.array_equal(out_vectors[0], vectors[0])
    assert np.array_equal(out_vectors[1], np.array([9.0, 9.0]))
    assert np.array_equal(out_vectors[2], np.array([9.0, 9.0]))
    assert np.array_equal(out_vectors[3], vectors[2])
