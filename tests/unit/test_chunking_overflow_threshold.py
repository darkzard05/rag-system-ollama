"""페이지 오버플로우 임계치 회귀 테스트.

일반적인 A4 1페이지(1,200~1,800자)가 하드코딩된 750자 임계치 때문에
고비용 문장 단위 시맨틱 청킹으로 오판정되는 결함을 방지한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import core.chunking as chunking_mod  # noqa: E402
from core.chunking import split_documents  # noqa: E402

STANDARD_PAGE_CHARS = 1200


def _page(index: int, chars: int = STANDARD_PAGE_CHARS) -> Document:
    return Document(
        page_content="단락 내용입니다. " * (chars // 10),
        metadata={"is_already_chunked": True, "page": index + 1},
    )


def _patch_config(monkeypatch: pytest.MonkeyPatch, threshold: int) -> None:
    monkeypatch.setattr(
        chunking_mod,
        "TEXT_SPLITTER_CONFIG",
        {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "page_overflow_threshold_chars": threshold,
        },
    )


@pytest.mark.asyncio
async def test_standard_pages_should_not_trigger_expensive_overflow(monkeypatch):
    """1페이지 1,200자 문서는 하위 분할 없이 원본 페이지 구조를 유지해야 한다."""
    docs = [_page(i) for i in range(5)]
    assert all(len(d.page_content) > 750 for d in docs)

    _patch_config(monkeypatch, threshold=2000)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": True})
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
async def test_legacy_750_threshold_is_no_longer_used(monkeypatch):
    """750자 초과만으로는 더 이상 오버플로우로 판정하지 않는다."""
    docs = [_page(i) for i in range(3)]
    _patch_config(monkeypatch, threshold=2000)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": False})
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    splitter_cls = MagicMock()
    monkeypatch.setattr(chunking_mod, "_recursive_splitter_cls", splitter_cls)

    split_docs, _ = await split_documents(docs, embedder=None)

    assert len(split_docs) == 3
    splitter_cls.assert_not_called()


@pytest.mark.asyncio
async def test_page_over_threshold_is_still_sub_chunked(monkeypatch):
    """임계치를 넘는 페이지는 기존대로 하위 분할 대상이어야 한다."""
    docs = [_page(0, chars=2500)]
    _patch_config(monkeypatch, threshold=2000)
    monkeypatch.setattr(chunking_mod, "SEMANTIC_CHUNKER_CONFIG", {"enabled": False})
    monkeypatch.setattr(
        chunking_mod.SessionManager, "add_status_log", lambda *a, **k: None
    )

    splitter_inst = MagicMock()
    splitter_inst.split_documents.return_value = [
        Document(page_content="part-1"),
        Document(page_content="part-2"),
        Document(page_content="part-3"),
    ]
    splitter_cls = MagicMock()
    splitter_cls.return_value.return_value = splitter_inst
    monkeypatch.setattr(chunking_mod, "_recursive_splitter_cls", splitter_cls)

    split_docs, _ = await split_documents(docs, embedder=None)

    assert len(split_docs) == 3
    splitter_cls.assert_called_once()
