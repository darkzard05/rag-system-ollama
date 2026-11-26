"""A regenerated final answer with no citations must clear the old ones.

The first generation pass can emit citations; a retry that fails JSON
recovery replaces the body with an empty citation list. Both streaming
consumers must drop the earlier citations instead of attaching them to the
new answer.
"""

from unittest.mock import patch

from api.stream_pipeline import StreamChunk, chunk_to_stream_events
from core.session import SessionManager
from ui.components.streaming import consume_stream_into_message
from ui.components.streaming_core import _AUX_STATE_KEY, _content_generator

_FIRST = StreamChunk(content="first pass", citations=[{"doc_id": "doc-1"}])
_REPLACE = StreamChunk(content="final answer", citations=[], is_replace=True)


def test_chunk_to_stream_events_emits_empty_citations_on_replace() -> None:
    events = chunk_to_stream_events(_REPLACE)
    citations = [event for event in events if event.type == "citations"]
    assert len(citations) == 1
    assert citations[0].payload == {"citations": []}


def test_consume_stream_into_message_clears_citations_on_replace() -> None:
    sid = "replace-clears-consume"
    SessionManager.reset_all_state(sid)
    import ui.components.streaming as streaming_mod

    with patch.object(
        streaming_mod, "stream_chunks", return_value=iter([_FIRST, _REPLACE])
    ):
        message = consume_stream_into_message(sid, "q", "m", msg_id="mid-clear")

    assert message is not None
    assert message["citations"] == []


def test_content_generator_clears_aux_citations_on_replace() -> None:
    sid = "replace-clears-generator"
    SessionManager.reset_all_state(sid)
    import ui.components.streaming_core as streaming_core_mod

    with patch.object(
        streaming_core_mod, "stream_chunks", return_value=iter([_FIRST, _REPLACE])
    ):
        list(_content_generator("q", "m", sid, "mid-aux-clear"))

    aux = SessionManager.get(_AUX_STATE_KEY, {}, session_id=sid)
    assert aux["citations"] == []
