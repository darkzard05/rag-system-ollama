"""TASK 5 Wave2 — FAISS lean metadata (zero-coords invariant).

FAISS docstore에는 좌표 블롭이 절대 저장되지 않아야 합니다. 린 문서는 허용
키 집합의 부분집합이어야 하고, 인덱싱 시 유입된 좌표 키는 저장 전에 제거
되지만 라우팅 키(file_hash/file_path/page/bbox/has_coordinates)는
하이드레이터를 위해 보존됩니다.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from core.retriever_factory import (
    FAISS_COORD_KEYS,
    FAISS_LEAN_METADATA_KEYS,
    create_vector_store,
    is_faiss_lean,
    strip_coord_metadata,
)


@pytest.fixture
def mock_session_manager(monkeypatch):
    """SessionManager.add_status_log를 모킹해 실제 세션 의존성을 제거합니다."""
    import core.retriever_factory as rf

    monkeypatch.setattr(rf, "SessionManager", MagicMock())


def _fat_docs() -> list[Document]:
    """좌표 블롭 + 라우팅 키를 모두 가진 인덱싱 입력 문서."""
    return [
        Document(
            page_content=f"chunk content number {i} for lean testing",
            metadata={
                "source": "report.pdf",
                "file_path": "C:/docs/report.pdf",
                "file_hash": "abc123",
                "page": i + 1,
                "total_pages": 3,
                "current_section": "Method",
                "has_coordinates": True,
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "word_coords": [(1.0, 2.0, 3.0, 4.0, "w")],
                "page_coords": {1: [(1.0, 2.0, 3.0, 4.0, "w")]},
                "coords": [(1.0, 2.0, 3.0, 4.0, "w")],
            },
        )
        for i in range(3)
    ]


def _vectors(n=3, dim=8):
    rng = np.random.default_rng(7)
    return [np.asarray(v, dtype="float32") for v in rng.random((n, dim))]


def _stored_metadatas(store) -> list[dict]:
    return [doc.metadata for doc in store.docstore._dict.values()]


def test_lean_allowlist_contract():
    """Given 키 집합 / When 린 판정 / Then 허용 집합의 부분집합일 때만 True다."""
    assert {
        "page",
        "current_section",
        "rerank_score",
        "score",
        "doc_id",
    } == FAISS_LEAN_METADATA_KEYS
    assert is_faiss_lean({"page": 1, "current_section": "A"}) is True
    assert is_faiss_lean({}) is True
    assert is_faiss_lean({"page": 1, "file_path": "x"}) is False
    assert is_faiss_lean({"page": 1, "word_coords": []}) is False


def test_strip_coord_metadata_keeps_routing_keys():
    """Given 좌표+라우팅 혼합 메타 / When 스트립 / Then 블롭만 제거되고
    원본은 불변·라우팅 키는 보존된다."""
    meta = _fat_docs()[0].metadata
    stripped = strip_coord_metadata(meta)
    assert set(stripped.keys()) & FAISS_COORD_KEYS == set()
    for key in (
        "file_path",
        "file_hash",
        "page",
        "bbox",
        "has_coordinates",
        "current_section",
    ):
        assert stripped[key] == meta[key]
    assert set(meta.keys()) & FAISS_COORD_KEYS != set()


def test_faiss_store_never_persists_coord_keys(mock_session_manager):
    """Given 좌표 블롭 포함 문서 / When FAISS 인덱싱 / Then 저장 메타데이터에
    좌표 키가 전혀 없고 라우팅 키·본문은 보존된다."""
    docs = _fat_docs()
    store = create_vector_store(docs, MagicMock(), vectors=_vectors())
    stored = _stored_metadatas(store)
    assert len(stored) == 3
    for meta in stored:
        assert set(meta.keys()) & FAISS_COORD_KEYS == set()
        assert meta["file_hash"] == "abc123"
        assert meta["file_path"] == "C:/docs/report.pdf"
        assert meta["has_coordinates"] is True
        assert meta["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert [d.page_content for d in store.docstore._dict.values()] == [
        d.page_content for d in docs
    ]
    for doc in docs:
        assert set(doc.metadata.keys()) & FAISS_COORD_KEYS != set()


def test_faiss_stored_lean_docs_are_subset(mock_session_manager):
    """Given 린 키만 가진 문서 / When FAISS 인덱싱 / Then 저장 키는 허용
    집합의 부분집합이고 좌표 키는 없다."""
    docs = [
        Document(
            page_content=f"lean chunk {i}",
            metadata={"page": i + 1, "current_section": "Intro", "doc_id": f"d{i}"},
        )
        for i in range(3)
    ]
    store = create_vector_store(docs, MagicMock(), vectors=_vectors())
    for meta in _stored_metadatas(store):
        assert is_faiss_lean(meta) is True
        assert set(meta.keys()) & FAISS_COORD_KEYS == set()
