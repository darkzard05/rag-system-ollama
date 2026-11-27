"""The model selector must store Ollama model ids, never display labels."""

from unittest.mock import patch

import pytest

from main import on_model_change
from ui.components.sidebar import format_llm_option, resolve_model_option

MODEL = "qwen3:4b-instruct-2507-q4_K_M"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (MODEL, MODEL),
        (format_llm_option(MODEL), MODEL),
        (None, ""),
        ("", ""),
    ],
)
def test_resolve_model_option(value: object, expected: str) -> None:
    assert resolve_model_option(value) == expected


def test_on_model_change_stores_raw_id_from_formatted_label() -> None:
    label = format_llm_option(MODEL)
    with (
        patch("main.st.session_state", {"model_selector": label}),
        patch(
            "core.session.SessionManager.get",
            return_value=None,
        ),
        patch("core.session.SessionManager.set") as set_mock,
    ):
        on_model_change()

    set_mock.assert_called_once_with("last_selected_model", MODEL)


def test_on_model_change_ignores_locale_only_refresh() -> None:
    """A locale switch re-serializes the same model into a new label."""
    label = format_llm_option(MODEL)
    with (
        patch("main.st.session_state", {"model_selector": label}),
        patch(
            "core.session.SessionManager.get",
            return_value=MODEL,
        ),
        patch("core.session.SessionManager.set") as set_mock,
    ):
        on_model_change()

    set_mock.assert_not_called()
