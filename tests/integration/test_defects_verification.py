"""
Verification tests for identified defects.
"""

import asyncio

import pytest
from src.core.session import SessionManager


@pytest.mark.asyncio
async def test_p0_2_session_sync_concurrency(session_context):
    """P0-2: 세션 동기화 및 스레드 안전성 검증"""

    async def worker(i):
        SessionManager.set(f"key_{i}", f"val_{i}", session_id=session_context)
        return SessionManager.get(f"key_{i}", session_id=session_context)

    results = await asyncio.gather(*(worker(i) for i in range(100)))

    for i, res in enumerate(results):
        assert res == f"val_{i}"


@pytest.mark.asyncio
async def test_use_llm_pins_actual_model():
    """결함-05: 비기본 LLM 모델 사용 시 추론 중 실제 모델명이 pin되는지 검증"""
    from common.config import DEFAULT_OLLAMA_MODEL
    from core.graph._graph_core import _safe_invoke
    from core.resource_manager import get_resource_manager

    coord = get_resource_manager()
    target_model = "custom-llm-model:latest"

    class MockLLM:
        model = target_model

        async def ainvoke(self, prompt, config=None):
            pinned = coord.models.is_pinned(f"llm_{target_model}")
            print(f"DEBUG pinned={pinned}, pinned_keys={coord.models._pinned_keys}")
            assert pinned
            return "mock-response"

    llm = MockLLM()
    await _safe_invoke(llm, "test prompt", {})

    assert not coord.models.is_pinned(f"llm_{target_model}")
