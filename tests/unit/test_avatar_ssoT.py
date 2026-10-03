"""Avatar SSoT guard: role->avatar mapping lives only in ``role_avatar``.

Covers the chat.py:79 ternary removal (AVATARS inline branch -> role_avatar
call) plus the common.py dead-key deletion (document/building).
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

import ui.components.chat as chat_module
from ui.components.chat import render_message
from ui.components.common import AVATARS, role_avatar


def _render_and_avatar(role: str) -> MagicMock:
    """Render one message and return the mocked ``st`` for avatar asserts."""
    with (
        patch("ui.components.chat.st") as mock_st,
        patch("ui.components.chat_references.st", mock_st),
    ):
        mock_st.container.return_value.__enter__.return_value = mock_st
        mock_st.chat_message.return_value.__enter__.return_value = mock_st
        mock_st.popover.return_value.__enter__.return_value = mock_st
        mock_st.expander.return_value.__enter__.return_value = mock_st
        mock_st.columns.return_value = [mock_st]
        render_message(role=role, content="hello", wrap_in_container=True)
    return mock_st


@pytest.mark.parametrize("role", ["assistant", "user", "tool"])
def test_avatar_matches_ssoT_accessor(role: str) -> None:
    """Equivalence guard: st.chat_message avatar == role_avatar(role)."""
    mock_st = _render_and_avatar(role)
    assert mock_st.chat_message.called, "wrap_in_container must use chat_message"
    _args, kwargs = mock_st.chat_message.call_args
    assert kwargs.get("avatar") == role_avatar(role)


def test_avatars_dict_has_only_live_keys() -> None:
    """Dead keys removed; live glyphs byte-identical."""
    assert "document" not in AVATARS
    assert "building" not in AVATARS
    assert AVATARS["assistant"] == "🤖"
    assert AVATARS["user"] == "👤"
    assert AVATARS["error"] == "❌"


def test_role_avatar_falls_back_to_user_glyph() -> None:
    """Unknown roles fall back to the user glyph."""
    assert role_avatar("tool") == "👤"
    assert role_avatar("nonsense-role") == "👤"


def test_role_avatar_is_used_by_chat_module() -> None:
    """Dead-code guard: chat.py must call the SSoT accessor."""
    src = inspect.getsource(chat_module)
    assert "role_avatar(role)" in src
