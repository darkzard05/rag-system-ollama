"""F4: 이벤트 루프 차단 제거 검증 테스트.

무거운 sync 작업이 asyncio 이벤트 루프를 차단하지 않고 워커 스레드로
오프로드되는지 heartbeat 패턴으로 검증합니다. heartbeat가 무거운 작업
실행 중에도 계속 틱할 수 있다면(>= 2회) 루프가 자유로움을 의미합니다.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

from langchain_core.documents import Document

from core.document_processor import load_pdf_docs
from core.resource_manager import ResourceCoordinator
from core.semantic_chunker import EmbeddingBasedSemanticChunker

_HEARTBEAT_INTERVAL = 0.02
_SLOW_WORK_SLEEP = 0.5
_MIN_TICKS = 2


async def _collect_ticks(coro) -> list[int]:
    """coro가 실행되는 동안 heartbeat 틱을 수집합니다."""
    ticks: list[int] = []
    stop = threading.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            ticks.append(1)
            await asyncio.sleep(_HEARTBEAT_INTERVAL)

    task = asyncio.create_task(coro)
    hb = asyncio.create_task(heartbeat())
    await task
    stop.set()
    await hb
    return ticks


async def test_load_pdf_docs_does_not_block_loop(tmp_path, monkeypatch):
    """load_pdf_docs의 PyMuPDF4LLM 추출이 루프를 차단하지 않아야 합니다."""
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 3
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_doc
    monkeypatch.setattr("core.document_processor.open_pdf_document", lambda p: mock_cm)

    def slow_to_markdown(doc, **kwargs):
        time.sleep(_SLOW_WORK_SLEEP)
        return [
            {
                "text": "hello",
                "page": 1,
                "page_num": 1,
                "current_section": "h1",
                "has_coords": False,
                "bbox": None,
                "tables": [],
                "words": [],
            }
        ]

    monkeypatch.setattr("pymupdf4llm.to_markdown", slow_to_markdown)

    ticks = await _collect_ticks(load_pdf_docs(str(pdf_path), "doc.pdf"))

    assert len(ticks) >= _MIN_TICKS, (
        f"load_pdf_docs가 이벤트 루프를 차단했습니다 (heartbeat {len(ticks)}회)"
    )


async def test_split_documents_does_not_block_loop(monkeypatch):
    """split_documents의 메타데이터 추출 executor.map 대기가 루프를 차단하지 않아야 합니다."""

    class _MockEmbedder:
        def embed_documents(self, texts):
            return [[0.1] for _ in texts]

        def embed_query(self, text):
            return [0.1]

    chunker = EmbeddingBasedSemanticChunker(embedder=_MockEmbedder())

    async def fake_split_text(text: str) -> list[dict]:
        return [
            {"text": "a", "vector": [0.1], "start": 0, "end": 1},
            {"text": "b", "vector": [0.2], "start": 2, "end": 3},
        ]

    monkeypatch.setattr(chunker, "split_text", fake_split_text)

    def slow_extract(chunk: dict, doc_ranges: list[dict]) -> dict:
        time.sleep(_SLOW_WORK_SLEEP)
        return {"page": 1}

    monkeypatch.setattr(chunker, "_extract_metadata_for_chunk", slow_extract)

    ticks = await _collect_ticks(
        chunker.split_documents([Document(page_content="hello world")])
    )

    assert len(ticks) >= _MIN_TICKS, (
        f"split_documents가 이벤트 루프를 차단했습니다 (heartbeat {len(ticks)}회)"
    )


async def test_get_or_build_does_not_block_loop():
    """get_or_build의 sync build_fn 호출이 루프를 차단하지 않아야 합니다."""
    mgr = ResourceCoordinator()
    pool = mgr.models
    await pool.remove("test-key")

    def slow_build() -> str:
        time.sleep(_SLOW_WORK_SLEEP)
        return "built"

    ticks = await _collect_ticks(mgr.get_or_build(pool, "test-key", slow_build))

    assert len(ticks) >= _MIN_TICKS, (
        f"get_or_build가 이벤트 루프를 차단했습니다 (heartbeat {len(ticks)}회)"
    )
