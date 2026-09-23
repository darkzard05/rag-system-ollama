"""SS silent-stream RED proof: pre-final "" yields no visible output.

Covers (read-only, no src/ edits):
- _generate.py 160-280/458-471: graph_status dispatch + response_chunk gating
  (``if (content_chunk or thought_chunk)`` — empty pre-final emits nothing).
- extractors 206-217: ``_FinalAnswerExtractor.feed`` returns "" pre-key;
  caller must ``if delta:``-guard (write_stream filters empty).
- state 169-176: ``_content_generator`` yields delta only ``if delta:``.
- runtime 271-279: ``stream_content`` yields delta only ``if delta:``.
- manager 486-500: ``add_status_log(..., add_to_chat=False)`` default.

Grep (2026-09-19): ``status_logs`` consumers = ``main.py``
(progress read), ``manager.py`` (store), ``rag_core.py:280`` (getter);
``_on_status`` def/use = ``chat.py:551/605`` only; ``_report_status`` =
``streaming_state.py:136/163/168`` only. Zero UI-timeline consumers —
pre-final window is silent (SS).

Contract (post-consolidation): pre-final "" yields the src placeholder
(``streaming_core._STATUS_PLACEHOLDER``, currently "" — a silent keep-alive
yield so ``st.write_stream`` stays live) exactly once, and the same value is
reported once via ``on_status`` for the caption slot. No new literals: the
constant is imported from src. ``stream_chunks`` mocks must target
``streaming_core`` (``streaming_runtime``/``streaming_state`` are re-export
stubs since the Phase 2B consolidation, so patching them is a no-op).
``add_status_log`` no-pollution guard documents the MUST NOT chat-pollute
constraint.
"""

from __future__ import annotations

import inspect
import os
from typing import Any
from unittest.mock import MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import ui.components.streaming_core as streaming_core_mod  # noqa: E402
import ui.components.streaming_runtime as runtime_mod  # noqa: E402
import ui.components.streaming_state as state_mod  # noqa: E402
from api.streaming_handler import StreamChunk  # noqa: E402
from core.session import SessionManager  # noqa: E402
from ui.components.streaming_core import _STATUS_PLACEHOLDER  # noqa: E402

# Pre-final raw_json fragment: key "final_answer" not yet arrived ->
# _FinalAnswerExtractor.feed returns "" (extractors 206-217).
_PRE_FINAL_CHUNK = StreamChunk(
    content='{"reasoning":"검토 중","final_ans', raw_json=True
)


class _FakeSessionManager(MagicMock):
    """Minimal SessionManager replica (get False / set no-op)."""

    _store: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._store = {}

    @classmethod
    def get(
        cls,
        key: str,
        default: Any = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return cls._store.get(key, default)

    @classmethod
    def set(
        cls, key: str, value: Any, session_id: str | None = None, **kwargs: Any
    ) -> None:
        cls._store[key] = value


def test_pre_final_empty_yields_status_placeholder() -> None:
    """Pre-final "" yields the src placeholder once (silent keep-alive).

    ``stream_content`` (runtime 271-279) filters via ``if delta:`` so the
    pre-final fragment yields no text; the trailing placeholder keeps
    ``st.write_stream`` live without inventing new literals.
    """
    _FakeSessionManager.reset()
    with (
        patch.object(
            streaming_core_mod,
            "stream_chunks",
            return_value=iter([_PRE_FINAL_CHUNK]),
        ),
        patch.object(runtime_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(runtime_mod.stream_content("q", "m", "s"))

    assert out == [_STATUS_PLACEHOLDER]


def test_add_status_log_keeps_add_to_chat_false_no_pollution() -> None:
    """``add_status_log`` default keeps add_to_chat=False, no chat pollution."""
    sig = inspect.signature(SessionManager.add_status_log)
    assert sig.parameters["add_to_chat"].default is False

    sid = "test-generate-placeholder-guard"
    SessionManager.init_session(sid)
    before = len(SessionManager.get_messages(session_id=sid))
    with patch.object(SessionManager, "add_message", autospec=True) as add_message_mock:
        SessionManager.add_status_log("답변 논리 설계 및 생성 시작", session_id=sid)
    add_message_mock.assert_not_called()
    after = len(SessionManager.get_messages(session_id=sid))
    assert after == before


def test_on_status_to_caption_contract_for_pre_final() -> None:
    """Pre-final silence still reports the placeholder once for the caption.

    ``_content_generator`` (state 169-176) reports the src
    placeholder for pure pre-final chunks ("" / raw_json pre-key) so the
    chat ``_on_status -> status_ph.caption`` contract has a value to render.
    """
    _FakeSessionManager.reset()
    reported: list[str] = []

    def _record(text: str, elapsed: float) -> None:
        reported.append(text)

    pre_final_empty = StreamChunk(content="", raw_json=True)
    with (
        patch.object(
            streaming_core_mod,
            "stream_chunks",
            return_value=iter([pre_final_empty]),
        ),
        patch.object(state_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(
            state_mod._content_generator("q", "m", "s", "mid", on_status=_record)
        )

    # Placeholder must be visible output, not silent.
    assert out == [_STATUS_PLACEHOLDER]
    # ... and reported via on_status for the caption slot.
    assert reported == [_STATUS_PLACEHOLDER]
    status_ph = MagicMock()

    def _on_status(status_text: str, elapsed_sec: float) -> None:
        status_ph.caption(status_text)

    for text in reported:
        _on_status(text, 0.0)
    status_ph.caption.assert_called_once_with(_STATUS_PLACEHOLDER)
