"""P1 회귀 테스트: 벡터 캐시 load/save는 이벤트 루프를 블로킹하지 않는다.

`src/core/pipeline_builder.py`의 캐시 히트(`cache.load`)·저장(`cache.save`)
호출은 동기 파일/FAISS/BM25 I/O이므로 `asyncio.to_thread`로 오프로드해야 한다.
- 가드 1: 소스 패턴 고정 (향후 직접 호출 회귀 시 실패).
- 가드 2: 블로킹 sleep을 to_thread로 감싸면 하트비트 태스크가 계속 틱한다
  (직접 호출이면 하트비트 간격이 블로킹 시간만큼 벌어진다).
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

import core.pipeline_builder as pb_mod


def test_cache_load_save_offloaded_to_thread() -> None:
    """소스 가드: load/save 직접 호출 회귀 방지."""
    src = inspect.getsource(pb_mod)
    assert "await asyncio.to_thread(cache.load" in src, (
        "cache.load must be offloaded via asyncio.to_thread"
    )
    assert "cache.save, doc_splits, vector_store, bm25_retriever" in src, (
        "cache.save must be offloaded via asyncio.to_thread"
    )
    # 직접 동기 호출 패턴이 남아 있으면 안 된다 (들여쓰기 무관, 행 단위 검사).
    direct_calls = [
        line.strip()
        for line in src.splitlines()
        if line.strip()
        in {
            "cache_data = cache.load(embedder)",
            "cache.save(doc_splits, vector_store, bm25_retriever)",
        }
    ]
    assert direct_calls == [], f"blocking direct cache I/O remains: {direct_calls}"


@pytest.mark.asyncio
async def test_to_thread_keeps_loop_responsive() -> None:
    """행위 가드: 0.5s 블로킹 작업을 offload하면 하트비트가 계속 틱한다."""

    def _blocking_load() -> str:
        time.sleep(0.5)
        return "cache-hit"

    beats: list[float] = []
    stop = False

    async def _heartbeat() -> None:
        while not stop:
            beats.append(time.monotonic())
            await asyncio.sleep(0.02)

    hb = asyncio.create_task(_heartbeat())
    try:
        result = await asyncio.to_thread(_blocking_load)
    finally:
        stop = True
        await hb

    assert result == "cache-hit"
    assert len(beats) >= 5, f"heartbeat starved during offload: {len(beats)} beats"
    gaps = [b - a for a, b in zip(beats, beats[1:], strict=False)]
    assert max(gaps) < 0.3, (
        f"event loop stalled during offload: max_gap={max(gaps):.3f}s"
    )
