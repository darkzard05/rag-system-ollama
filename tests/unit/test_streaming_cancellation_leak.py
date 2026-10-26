import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.exceptions import LLMInferenceError
from common.stream_worker import (
    active_stream_count,
    release_stream_slot,
)
from core.resource_manager import get_resource_manager
from ui.components.streaming_core import _acquire_slot_async


@pytest.mark.asyncio
async def test_acquire_slot_async_normal():
    """정상 획득 테스트."""
    ok = await _acquire_slot_async(timeout=0.1)
    if ok:
        release_stream_slot()
    assert isinstance(ok, bool)


@pytest.mark.asyncio
async def test_acquire_slot_async_cancellation_leak():
    """RED 테스트: _acquire_slot_async가 취소되었을 때 지연 획득된 슬롯이 누수되지 않고 반납되는지 증명."""
    initial_slots = active_stream_count()

    task = asyncio.create_task(_acquire_slot_async(timeout=1.0))
    await asyncio.sleep(0.01)  # 스레드 진입 유도
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(1.2)

    assert active_stream_count() == initial_slots, (
        "태스크 취소 시 지연 획득된 슬롯이 누수되었습니다!"
    )


@pytest.mark.asyncio
async def test_acquire_inference_lock_cancellation_preserves_semaphore():
    """RED 테스트: acquire_inference_lock 대기 중 취소 시 세마포어 퍼밋 누수 방지 증명."""
    coord = get_resource_manager()
    coord.reset()

    await coord.acquire_inference_lock(timeout=1.0)

    task = asyncio.create_task(coord.acquire_inference_lock(timeout=1.0))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    coord.release_inference_lock()

    try:
        await asyncio.wait_for(coord.acquire_inference_lock(timeout=0.5), timeout=1.0)
        coord.release_inference_lock()
    except (asyncio.TimeoutError, LLMInferenceError):
        pytest.fail("추론 세마포어가 취소 레이스로 인해 고갈(Deadlock)되었습니다!")


@pytest.mark.asyncio
async def test_astream_unconsumed_generator_pin_leak():
    """RED 테스트: astream() 호출 후 이터레이션 없이 폐기될 때 리트리버 Pin 카운트 누수 방지 증명."""
    from core.rag_core import RAGSystem

    coord = get_resource_manager()
    coord.reset()

    file_hash = "dummy_test_hash_12345"
    with (
        patch("core.rag_core.prepare_query_config_or_build", new_callable=AsyncMock),
        patch("core.session.SessionManager.get", return_value=file_hash),
        patch.object(
            RAGSystem, "_get_rag_engine", new_callable=AsyncMock
        ) as mock_get_eng,
    ):
        mock_get_eng.return_value = MagicMock()
        coord.retrievers.pin(file_hash)
        assert coord.retrievers.is_pinned(file_hash) is True
        coord.retrievers.unpin(file_hash)
        assert coord.retrievers.is_pinned(file_hash) is False

        rag_sys = RAGSystem(session_id="test_sess")
        gen = await rag_sys.astream("test query")

        del gen
        import gc

        gc.collect()

        assert coord.retrievers.is_pinned(file_hash) is False, (
            "미소비 제너레이터로 인해 리트리버 Pin이 고착되었습니다!"
        )
