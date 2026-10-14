"""OPT-2 재현 테스트: SSE ``is_disconnected`` 샘플링.

실패 조건 (수정 전): 실제 ``stream_query_rag`` 루프가 토큰 청크마다
``await fastapi_request.is_disconnected()`` 를 호출해 호출 횟수 == 청크 수.

성공 조건 (수정 후): 호출 횟수 <= ceil(N / DISCONNECT_CHECK_EVERY) + 1 이고,
SSE 계약(배치 무손실, ``end`` 정확히 1회)이 유지된다.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routes_chat as routes_chat
from api.schemas import QueryRequest
from api.stream_pipeline import StreamChunk

N_CHUNKS = 32


def _make_chunks(n: int) -> list[StreamChunk]:
    return [
        StreamChunk(content=f"tok{i} ", chunk_index=i, token_count=1) for i in range(n)
    ]


class _FakeHandler:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = chunks

    async def stream_graph_events(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        for ch in self._chunks:
            yield ch


def _make_fake_srv(chunks: list[StreamChunk]) -> SimpleNamespace:
    fake_session = SimpleNamespace(
        init_session=lambda **kwargs: None,
        set_session_id=lambda sid: None,
        get=lambda key, session_id=None: "doc.pdf",
        add_status_log=lambda *args, **kwargs: None,
    )

    class _FakeRAG:
        def __init__(self, session_id: str = "") -> None:
            self.session_id = session_id

        async def astream(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            return object()

    return SimpleNamespace(SessionManager=fake_session, RAGSystem=_FakeRAG)


async def _drain_stream(n: int) -> tuple[SimpleNamespace, str, MagicMock]:
    chunks = _make_chunks(n)
    fake_request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    with (
        patch.object(
            routes_chat, "_app_server_module", return_value=_make_fake_srv(chunks)
        ),
        patch.object(
            routes_chat, "get_streaming_handler", return_value=_FakeHandler(chunks)
        ),
        patch.object(routes_chat, "get_adaptive_controller", return_value=MagicMock()),
    ):
        resp = await routes_chat.stream_query_rag(
            QueryRequest(query="hello", session_id="opt2-repro-sid"),
            fake_request,  # type: ignore[arg-type]
            "user-1",
            "default",
        )
        parts: list[str] = []
        async for part in resp.body_iterator:
            parts.append(part if isinstance(part, str) else part.decode("utf-8"))
    return fake_request, "".join(parts), fake_request.is_disconnected


@pytest.mark.asyncio
async def test_disconnect_check_is_sampled() -> None:
    """연결 끊김 확인이 매 청크가 아닌 샘플링 간격으로 수행되어야 한다."""
    _, body, is_disconnected = await _drain_stream(N_CHUNKS)
    interval = getattr(routes_chat, "DISCONNECT_CHECK_EVERY", 1)
    assert interval >= 8, (
        f"DISCONNECT_CHECK_EVERY={interval}: is_disconnected is still "
        "polled on (nearly) every chunk"
    )
    budget = math.ceil(N_CHUNKS / interval) + 1
    assert is_disconnected.call_count <= budget, (
        f"is_disconnected calls={is_disconnected.call_count} > budget={budget}: "
        "per-chunk polling still in place"
    )
    assert body.count("event: end") == 1, "SSE must terminate with exactly one end"
    for i in range(N_CHUNKS):
        assert f"tok{i} " in body, f"token chunk {i} lost from SSE output"
