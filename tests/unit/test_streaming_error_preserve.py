from __future__ import annotations

from unittest.mock import patch

import pytest

from core.session import SessionManager
from ui.components import streaming_core as sc


def test_content_generator_error_not_overwritten_by_finally():
    sid = "test_error_preserve_sid"
    SessionManager.reset_all_state(sid)

    # 스트리밍 도중 예외 발생 유도
    with patch.object(
        sc, "stream_chunks", side_effect=RuntimeError("Ollama connection lost")
    ):
        gen = sc._content_generator("q", "model", sid, "mid_err")
        with pytest.raises(RuntimeError):
            list(gen)

    # finally가 error 키를 지우거나 complete: True로 덮어쓰지 않았는지 검증
    aux = SessionManager.get("stream_active_aux", session_id=sid)
    assert aux is not None
    assert aux.get("error") == "Ollama connection lost", (
        f"Error was swallowed: {aux.get('error')}"
    )
    assert aux.get("complete") is False, "Failure state was marked as complete: True"
