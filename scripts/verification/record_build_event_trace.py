"""Record a real BuildEvent stream from the pipeline with a fake embedder.

Usage:
    python scripts/verification/record_build_event_trace.py

Writes the raw event sequence to reports/build_event_trace_<stamp>.json.
The trace is the input for the completion-indicator comparison harness.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from langchain_core.documents import Document  # noqa: E402

from core.build_state import BuildEvent  # noqa: E402
from core.pipeline_builder import PipelineBuilder  # noqa: E402


class FakeEmbedder:
    model = "trace-fake-embedder"
    model_name = "trace-fake-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.01] * 8


async def _fake_load_pdf_docs(
    file_path: str,
    file_name: str,
    on_progress: Any = None,
    **kwargs: Any,
) -> list[Document]:
    if on_progress is not None:
        try:
            on_progress(15, detail="Extracting...")
        except TypeError:
            on_progress(15)
    return [Document(page_content="Document analysis completion trace.")]


async def record() -> list[dict[str, Any]]:
    events: list[BuildEvent] = []
    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)
    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash_trace"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch("core.pipeline_builder.VectorStoreCache.save"),
        patch("core.pipeline_builder.load_pdf_docs", new=_fake_load_pdf_docs),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=([Document(page_content="c")], [])),
        ),
        patch("core.pipeline_builder.create_vector_store", return_value=MagicMock()),
        patch("core.pipeline_builder.create_bm25_retriever", return_value=MagicMock()),
        patch(
            "core.pipeline_builder.get_resource_manager",
            return_value=resource_manager,
        ),
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        builder = PipelineBuilder(session_id="trace-completion")
        await builder.build(
            file_path="trace.pdf",
            file_name="trace.pdf",
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            on_event=events.append,
        )
    return [
        {
            "phase": e.phase.value,
            "pct": e.pct,
            "detail_key": e.detail_key,
            "status": e.status,
            "ts": e.ts,
        }
        for e in events
    ]


def main() -> int:
    trace = asyncio.run(record())
    if not trace:
        print("[ERROR] no BuildEvent recorded")
        return 1
    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir, f"build_event_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, ensure_ascii=False)
    print(f"recorded {len(trace)} events -> {out}")
    for item in trace:
        print(f"  {item['phase']:<12} {item['pct']:>3}% {item['detail_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
