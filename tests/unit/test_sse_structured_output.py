"""SSE must emit the parsed final answer, not raw structured JSON."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routes_chat as routes_chat
from api.schemas import QueryRequest
from api.stream_pipeline import StreamChunk


class _FakeHandler:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = chunks

    async def stream_graph_events(self, *args: object, **kwargs: object):  # noqa: ANN201, ARG002
        for chunk in self._chunks:
            yield chunk


def _make_fake_srv() -> SimpleNamespace:
    fake_session = SimpleNamespace(
        init_session=lambda **kwargs: None,
        set_session_id=lambda sid: None,
        get=lambda key, session_id=None: "doc.pdf",
        add_status_log=lambda *args, **kwargs: None,
    )

    class _FakeRAG:
        def __init__(self, session_id: str = "") -> None:
            self.session_id = session_id

        async def astream(self, *args: object, **kwargs: object):  # noqa: ANN201, ARG002
            return object()

    return SimpleNamespace(SessionManager=fake_session, RAGSystem=_FakeRAG)


@pytest.mark.asyncio
async def test_sse_emits_parsed_final_answer() -> None:
    chunks = [
        StreamChunk(content='{"final_answer": "Hello ', raw_json=True, chunk_index=0),
        StreamChunk(content='world"}', raw_json=True, chunk_index=1, is_final=True),
    ]
    fake_request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    with (
        patch.object(routes_chat, "_app_server_module", return_value=_make_fake_srv()),
        patch.object(
            routes_chat, "get_streaming_handler", return_value=_FakeHandler(chunks)
        ),
        patch.object(routes_chat, "get_adaptive_controller", return_value=MagicMock()),
    ):
        response = await routes_chat.stream_query_rag(
            QueryRequest(query="hello", session_id="sse-structured-sid"),
            fake_request,  # type: ignore[arg-type]
            "user-1",
            "default",
        )
        parts: list[str] = []
        async for part in response.body_iterator:
            parts.append(part if isinstance(part, str) else part.decode("utf-8"))
        body = "".join(parts)

    message_contents = [
        json.loads(line.removeprefix("data: "))["content"]
        for line in body.splitlines()
        if line.startswith("data: {") and '"content"' in line
    ]
    assert "".join(message_contents) == "Hello world"
    assert body.count("event: end") == 1
