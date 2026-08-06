"""build_graph() 임계 구간 보호 테스트 (Bug B).

Bug B: ``build_graph``의 배타 락이 이중 확인(double-check)만 보호하고
``StateGraph`` 구성/컴파일은 락 밖에서 실행되어 동시 호출 시 중복 컴파일
(StateGraph가 두 번 구성됨) 위험이 있었습니다.

검증: (1) StateGraph 구성/컴파일 시점에 빌드 락이 반드시 점유되어 있어야 하며
(2) 동시 ``build_graph()`` 호출은 그래프를 정확히 1회만 컴파일해야 합니다.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import core.graph_builder as gb


@pytest.mark.asyncio
async def test_build_graph_constructs_under_lock():
    """StateGraph 구성/컴파일은 빌드 락을 점유한 상태에서 실행되어야 합니다."""
    probe = SimpleNamespace(count=0, lock_held=None)
    real = gb.StateGraph

    class Probe(real):
        def __init__(self, *args, **kwargs):
            probe.count += 1
            probe.lock_held = gb._graph_cache.get_lock().locked()
            super().__init__(*args, **kwargs)

    gb.invalidate_graph_cache()
    with patch.object(gb, "StateGraph", Probe):
        graph = await gb.build_graph()
    gb.invalidate_graph_cache()

    assert probe.count == 1
    assert probe.lock_held is True, (
        "StateGraph construction ran while the build lock was NOT held → "
        "construction + compile execute OUTSIDE the lock → two concurrent "
        "build_graph() calls could both construct/compile (double compile). "
        "The critical section must live inside the single `async with lock:`."
    )
    assert graph is not None


@pytest.mark.asyncio
async def test_concurrent_build_graph_compiles_exactly_once():
    """그래프 캐시가 비어 있을 때 동시 build_graph() 두 번은 1회만 컴파일해야 합니다."""
    state = {"count": 0}
    real = gb.StateGraph

    class Counting(real):
        def __init__(self, *args, **kwargs):
            state["count"] += 1
            super().__init__(*args, **kwargs)

    gb.invalidate_graph_cache()
    with patch.object(gb, "StateGraph", Counting):
        graph_a, graph_b = await asyncio.gather(gb.build_graph(), gb.build_graph())
    gb.invalidate_graph_cache()

    assert state["count"] == 1, (
        f"concurrent build_graph() calls constructed StateGraph {state['count']} times — "
        "expected exactly one construction; the second caller must reuse the cached graph."
    )
    assert graph_a is graph_b
