"""TASK 2 Wave2: ghost placeholder pre-render at documents arrival.

- Ghost appears when ``sources`` documents arrive (before citations/final).
- Final replaces ghost 1:1 keyed by ``msg_id`` (no duplication).
- Stale ghost (changed citations / mismatched msg_id) is discarded,
  never persisted.
- Ordering: ``[GHOST]`` timestamp precedes ``[PERF] ... persist``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from core.session import SessionManager


def _make_docs(n: int = 2) -> list[dict[str, Any]]:
    """Minimal dict-docs supported by doc_stable_id/get_doc_metadata."""
    return [
        {
            "page_content": f"ghost doc body {i} alpha beta gamma.",
            "metadata": {
                "doc_id": f"ghost-doc-{i}",
                "page": i + 1,
                "current_section": f"Ghost Section {i}",
            },
        }
        for i in range(n)
    ]


def _make_citations(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": d["metadata"]["doc_id"],
            "section": d["metadata"]["current_section"],
            "page": d["metadata"]["page"],
        }
        for d in docs
    ]


def _chunk(
    content: str = "",
    metadata: dict[str, Any] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=None,
        thought="",
        content=content,
        metadata=metadata,
        performance=None,
        citations=citations,
        raw_json=False,
    )


def test_ghost_snapshot_flag_on_sources() -> None:
    """Ghost snapshot appears when sources arrive, clears on citations."""
    from ui.components.streaming import consume_stream_into_message

    sid = "test-ghost-sources"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)

    docs = _make_docs(2)
    cits = _make_citations(docs)

    def fake_stream(query: str, model_name: str, session_id: str):
        yield _chunk(metadata={"documents": docs})
        yield _chunk(content="partial answer ")
        yield _chunk(citations=cits, content="done.")

    snapshots: list[dict[str, Any]] = []
    with patch("ui.components.streaming.stream_chunks", fake_stream):
        consume_stream_into_message(
            sid,
            "ghost q",
            "model",
            msg_id="msg-ghost-1",
            on_chunk=snapshots.append,
        )

    ghost_snaps = [s for s in snapshots if s.get("has_ghost") is True]
    assert ghost_snaps, "sources arrival must emit a ghost snapshot"
    first = ghost_snaps[0]
    assert first["documents"] == docs
    assert not first["citations"], "ghost means docs arrived, citations not yet"
    assert first["ghost_msg_id"] == "msg-ghost-1"
    assert first["ghost_ts"] is not None

    real_snaps = [s for s in snapshots if s.get("citations")]
    assert real_snaps, "citations chunk must be observed"
    assert all(s.get("has_ghost") is False for s in real_snaps), (
        "changed citations discard ghost"
    )


def test_ghost_skeleton_rows_carry_data_ghost_and_msg_id() -> None:
    """Generating skeleton renders placeholder rows, no widgets."""
    from ui.components import chat_references as refs

    fake_st = MagicMock()
    with patch.object(refs, "st", fake_st):
        ok = refs._render_references_content(
            "msg-ghost-2",
            _make_docs(2),
            citations=None,
            generating=True,
        )
    assert ok is True
    assert fake_st.button.call_count == 0, "skeleton must not create widgets"
    assert fake_st.markdown.call_count >= 1
    bodies = [str(c.args[0]) for c in fake_st.markdown.call_args_list]
    assert any("data-ghost" in b for b in bodies)
    assert any("msg-ghost-2" in b for b in bodies)
    assert any("search" in b.lower() for b in bodies), "searching badge rows"


def test_final_replaces_ghost_one_to_one() -> None:
    """Ghost rows and final rows match 1:1; final carries no ghost marks."""
    from ui.components import chat_references as refs

    docs = _make_docs(2)
    cits = _make_citations(docs)

    ghost_st = MagicMock()
    with patch.object(refs, "st", ghost_st):
        assert (
            refs._render_references_content(
                "msg-ghost-3",
                docs,
                citations=None,
                generating=True,
            )
            is True
        )
    ghost_rows = len(ghost_st.markdown.call_args_list)

    final_st = MagicMock()
    with patch.object(refs, "st", final_st):
        assert (
            refs._render_references_content(
                "msg-ghost-3",
                docs,
                citations=cits,
                generating=False,
            )
            is True
        )
    assert final_st.button.call_count == ghost_rows, "final replaces ghost 1:1"
    keys = [c.kwargs.get("key", "") for c in final_st.button.call_args_list]
    assert all(str(k).startswith("pop_doc_msg-ghost-3_") for k in keys)
    final_bodies = [str(c.args[0]) for c in final_st.markdown.call_args_list]
    assert not any("data-ghost" in b for b in final_bodies)


def test_stale_ghost_discarded_never_persists() -> None:
    """Stale ghost (other msg_id / changed citations) never persists."""
    from ui.components import chat as chat_mod
    from ui.components import chat_references as refs
    from ui.components.streaming import consume_stream_into_message

    # (a) mismatched msg_id is stale
    assert chat_mod._is_stale_ghost({"ghost_msg_id": "other"}, "msg-ghost-4") is True
    # (b) matching msg_id is live
    assert (
        chat_mod._is_stale_ghost({"ghost_msg_id": "msg-ghost-4"}, "msg-ghost-4")
        is False
    )
    # (c) legacy aux without ghost key is not stale (backward compatible)
    assert chat_mod._is_stale_ghost({"thought": "t"}, "msg-ghost-4") is False
    # (d) generating render WITH citations takes the real path (no ghost)
    docs = _make_docs(2)
    real_st = MagicMock()
    with patch.object(refs, "st", real_st):
        refs._render_references_content(
            "msg-ghost-4",
            docs,
            citations=_make_citations(docs),
            generating=True,
        )
    real_bodies = [str(c.args[0]) for c in real_st.markdown.call_args_list]
    assert not any("data-ghost" in b for b in real_bodies)

    # (e) persisted message carries no ghost keys
    sid = "test-ghost-stale"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    docs_e = _make_docs(1)

    def fake_stream(query: str, model_name: str, session_id: str):
        yield _chunk(metadata={"documents": docs_e})
        yield _chunk(content="final text")

    with patch("ui.components.streaming.stream_chunks", fake_stream):
        result = consume_stream_into_message(
            sid,
            "q",
            "model",
            msg_id="msg-ghost-5",
        )
    assert result is not None
    assert "has_ghost" not in result
    assert "ghost_msg_id" not in result
    assert "data-ghost" not in str(result.get("content", ""))


def test_ghost_timestamp_before_persist(
    caplog: Any,
) -> None:
    """Ghost log precedes persist log (perceived-latency ordering)."""
    from ui.components.streaming import consume_stream_into_message

    sid = "test-ghost-order"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    docs = _make_docs(2)

    def fake_stream(query: str, model_name: str, session_id: str):
        yield _chunk(metadata={"documents": docs})
        yield _chunk(content="answer text")

    with (
        patch("ui.components.streaming.stream_chunks", fake_stream),
        caplog.at_level(logging.INFO, logger="ui.components.streaming"),
    ):
        caplog.clear()
        consume_stream_into_message(sid, "q", "model", msg_id="msg-ghost-6")

    assert "[GHOST]" in caplog.text, "ghost must be visible in log timestamps"
    assert "[PERF] consume persist" in caplog.text
    assert caplog.text.find("[GHOST]") < caplog.text.find("[PERF] consume persist")
