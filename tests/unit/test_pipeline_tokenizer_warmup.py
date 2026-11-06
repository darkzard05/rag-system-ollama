"""PipelineBuilder 토크나이저 워밍업 회귀 테스트.

API/스크립트가 UI를 거치지 않고 ``PipelineBuilder`` 를 직접 호출해도
HF 토크나이저 콜드 로드가 텍스트 추출과 중첩되고, 청킹 전에 완료되어야 한다.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_pipeline_warms_tokenizer_before_split(monkeypatch):
    events: list[str] = []

    async def fake_warm() -> bool:
        events.append("warm_start")
        await asyncio.sleep(0.05)
        events.append("warm_done")
        return True

    async def fake_load(*args, **kwargs):
        events.append("extract_start")
        await asyncio.sleep(0.01)
        events.append("extract_done")
        return [Document(page_content="content")]

    async def fake_split(*args, **kwargs):
        events.append("split")
        return [Document(page_content="chunk")], []

    monkeypatch.setattr("core.pipeline_builder.is_test_env", lambda: False)
    monkeypatch.setattr(
        "core.pipeline_builder.warm_embedding_tokenizer_async", fake_warm
    )
    monkeypatch.setattr("core.pipeline_builder.load_pdf_docs", fake_load)
    monkeypatch.setattr("core.pipeline_builder.split_documents", fake_split)
    monkeypatch.setattr("core.pipeline_builder.ENABLE_VECTOR_CACHE", False)
    monkeypatch.setattr("core.pipeline_builder._warmup_faiss_import", lambda: None)
    monkeypatch.setattr(
        "core.pipeline_builder.create_vector_store", lambda *a, **k: "vs"
    )
    monkeypatch.setattr(
        "core.pipeline_builder.create_bm25_retriever", lambda *a, **k: "bm25"
    )

    builder = PipelineBuilder(session_id="tokenizer-warmup-test")
    monkeypatch.setattr(builder, "_register_and_finalize", AsyncMock(return_value=None))

    embedder = MagicMock()
    embedder.model = "fake-model"

    await builder.build(
        file_path="fake.pdf",
        file_name="fake.pdf",
        embedder=embedder,
        file_hash="hash-warmup",
    )

    assert "warm_done" in events
    assert events.index("warm_start") < events.index("extract_done")
    assert events.index("warm_done") < events.index("split")


@pytest.mark.asyncio
async def test_pipeline_continues_when_tokenizer_warmup_fails(monkeypatch):
    async def failing_warm() -> bool:
        raise RuntimeError("tokenizer unavailable")

    async def fake_load(*args, **kwargs):
        return [Document(page_content="content")]

    split_called = False

    async def fake_split(*args, **kwargs):
        nonlocal split_called
        split_called = True
        return [Document(page_content="chunk")], []

    monkeypatch.setattr("core.pipeline_builder.is_test_env", lambda: False)
    monkeypatch.setattr(
        "core.pipeline_builder.warm_embedding_tokenizer_async", failing_warm
    )
    monkeypatch.setattr("core.pipeline_builder.load_pdf_docs", fake_load)
    monkeypatch.setattr("core.pipeline_builder.split_documents", fake_split)
    monkeypatch.setattr("core.pipeline_builder.ENABLE_VECTOR_CACHE", False)
    monkeypatch.setattr("core.pipeline_builder._warmup_faiss_import", lambda: None)
    monkeypatch.setattr(
        "core.pipeline_builder.create_vector_store", lambda *a, **k: "vs"
    )
    monkeypatch.setattr(
        "core.pipeline_builder.create_bm25_retriever", lambda *a, **k: "bm25"
    )

    builder = PipelineBuilder(session_id="tokenizer-warmup-fail-test")
    monkeypatch.setattr(builder, "_register_and_finalize", AsyncMock(return_value=None))

    embedder = MagicMock()
    embedder.model = "fake-model"

    await builder.build(
        file_path="fake.pdf",
        file_name="fake.pdf",
        embedder=embedder,
        file_hash="hash-warmup-fail",
    )

    assert split_called is True
