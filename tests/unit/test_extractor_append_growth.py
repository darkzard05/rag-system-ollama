"""RED: _FinalAnswerExtractor must not copy the whole buffer per token.

Covers src/ui/components/streaming_core.py _FinalAnswerExtractor.feed
(``self._blob += content`` — O(n) copy per token, O(n^2) per answer).

T1 (structural, RED before fix): inputs accumulate in a chunk list.
T2 (scaling, RED before fix): 3x input growth costs <5x time (quadratic ~=9x).
T3/T4 (guards, GREEN before and after): delta correctness incl. escapes
split across chunk boundaries, closed-tail behavior, and _content_generator
long-run output equality.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import ui.components.streaming as streaming_mod  # noqa: E402
import ui.components.streaming_core as streaming_core_mod  # noqa: E402
from api.streaming_handler import StreamChunk  # noqa: E402


def test_extractor_accumulates_chunks_without_full_copy() -> None:
    """T1: fed inputs must be retained as an append-only chunk list."""
    extractor = streaming_core_mod._FinalAnswerExtractor()
    parts = ['{"final_answer":"', "hello ", "world"]
    for part in parts:
        extractor.feed(part)
    chunks = extractor._chunks  # type: ignore[attr-defined]
    assert isinstance(chunks, list)
    assert "".join(chunks) == "".join(parts)


def test_extractor_holds_no_unbounded_private_buffer() -> None:
    """T2: no private full-copy string may grow with the answer length.

    Deterministic proxy for the O(n^2) copy: after 7500 post-key chunks the
    legacy ``_blob`` holds ~360KB. Any retained contiguous buffer must stay
    bounded (8KB) — the chunk list from T1 is the only growth allowed.
    """
    extractor = streaming_core_mod._FinalAnswerExtractor()
    extractor.feed('{"final_answer":"')
    for _ in range(7500):
        extractor.feed("x" * 48)
    retained = getattr(extractor, "_blob", "")
    assert len(retained) <= 8192, f"unbounded copy buffer: {len(retained)} chars"


def test_extractor_delta_correctness_with_split_escapes() -> None:
    """T3a: escapes split across chunk boundaries decode exactly once."""
    extractor = streaming_core_mod._FinalAnswerExtractor()
    # Split in the middle of \\n and \\u0041 escape sequences.
    parts = ['{"final_answer":"a', "\\", "nB", "\\u00", "41C", '","citations":[]}']
    deltas = [extractor.feed(part) for part in parts]
    assert "".join(deltas) == "a\nB\u0041C"
    assert extractor.closed is True


def test_extractor_closed_ignores_trailing_chunks() -> None:
    """T3b: after close, further feed returns '' without rescanning."""
    extractor = streaming_core_mod._FinalAnswerExtractor()
    assert extractor.feed('{"final_answer":"hi","other":"x"}') == "hi"
    assert extractor.closed is True
    assert extractor.feed(',"confidence":0.9}') == ""


class _FakeSessionManager(MagicMock):
    _store: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._store = {}

    @classmethod
    def get(
        cls, key: str, default: Any = None, session_id: str | None = None, **kwargs: Any
    ) -> Any:
        return cls._store.get(key, default)

    @classmethod
    def set(
        cls, key: str, value: Any, session_id: str | None = None, **kwargs: Any
    ) -> None:
        cls._store[key] = value

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return []


def test_content_generator_long_plain_run_matches_join() -> None:
    """T4: 1500 plain chunks yield the exact join (accumulated refactor guard)."""
    _FakeSessionManager.reset()
    pieces = [f"tok{i:04d} " for i in range(1500)]
    chunks = [StreamChunk(content=piece) for piece in pieces]
    with (
        patch.object(streaming_core_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_core_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(streaming_mod._content_generator("q", "m", "s", "mid"))
    assert out == pieces
    assert "".join(out) == "".join(pieces)
