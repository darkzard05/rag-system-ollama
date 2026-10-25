"""SSE 스트리밍 중 클라이언트 절단 시 aclose() 동시성 위반 및 리트리버 핀 누수 방지 테스트 (TDD GREEN)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import api.routes_chat as routes_chat
from api.schemas import QueryRequest
from core.resource_manager import get_resource_manager


@pytest.mark.asyncio
async def test_sse_disconnect_aclose_race_and_pin_leak() -> None:
    """클라이언트 연결이 절단될 때 waiter 취소 및 aclose() 레이스 컨디션으로 인한

    RuntimeError 방지 및 리트리버 핀 해제 검증.
    """
    file_hash = "test_file_hash_leak_999"
    resource_manager = get_resource_manager()
    resource_manager.reset()

    mock_vs = AsyncMock()
    mock_vs.as_retriever.return_value = AsyncMock()
    await resource_manager.register_retrievers(file_hash, mock_vs, AsyncMock())

    # Pin the retriever initially as RAGSystem.astream does
    resource_manager.retrievers.pin(file_hash)
    assert resource_manager.retrievers.is_pinned(file_hash) is True

    fake_session = SimpleNamespace(
        init_session=lambda **kwargs: None,
        set_session_id=lambda sid: None,
        get=lambda key, session_id=None: file_hash
        if key == "last_uploaded_file_name"
        else (file_hash if key == "file_hash" else None),
        add_status_log=lambda *args, **kwargs: None,
    )

    class _FakeRAGSystem:
        def __init__(self, session_id: str = "") -> None:
            self.session_id = session_id

        async def astream(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            async def _consumer():
                try:
                    yield ("custom", {"content": "start"})
                    await asyncio.sleep(2.0)
                    yield ("custom", {"content": "later"})
                finally:
                    resource_manager.retrievers.unpin(file_hash)

            return _consumer()

    fake_srv = SimpleNamespace(
        SessionManager=fake_session,
        RAGSystem=_FakeRAGSystem,
    )

    # is_disconnected returns True on second check to trigger disconnect handling
    fake_request = SimpleNamespace(
        is_disconnected=AsyncMock(side_effect=[False, True, True, True])
    )

    with patch.object(routes_chat, "_app_server_module", return_value=fake_srv):
        resp = await routes_chat.stream_query_rag(
            QueryRequest(query="test query", session_id="test_leak_sid"),
            fake_request,  # type: ignore[arg-type]
            "user-1",
            "default",
        )

        # Fully consume stream body iterator to trigger event_generator and disconnect handling
        try:
            async for _ in resp.body_iterator:
                pass
        except Exception as e:
            pytest.fail(f"Stream consumer raised unexpected exception: {e}")

    # Verify that retriever pin was released and no RuntimeError occurred
    assert not resource_manager.retrievers.is_pinned(file_hash), (
        "Retriever pin leaked after SSE client disconnect due to aclose race!"
    )
