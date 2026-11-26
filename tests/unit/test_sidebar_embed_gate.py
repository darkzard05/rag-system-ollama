"""Embedding selector follows the same Ollama reachability gate as the LLM one."""

from __future__ import annotations

from unittest.mock import patch

from ui.components.sidebar import render_settings_content


def _select_kwargs(mock_st, key: str) -> dict:
    calls = [
        call
        for call in mock_st.selectbox.call_args_list
        if call.kwargs.get("key") == key
    ]
    assert len(calls) == 1, f"selectbox {key!r} rendered {len(calls)} times"
    return calls[0].kwargs


def test_embedding_selector_disabled_when_ollama_is_down() -> None:
    """Both model selectors must be locked together when Ollama is unreachable."""
    with (
        patch("ui.components.sidebar.st") as mock_st,
        patch(
            "core.model_catalog.get_filtered_models",
            return_value={"llm": ["llm-a"], "embedding": ["emb-a"]},
        ),
    ):
        mock_st.button.return_value = False
        render_settings_content(
            lambda *_args, **_kwargs: None,
            lambda *_args, **_kwargs: None,
            lambda *_args, **_kwargs: None,
            available_models=["llm-a", "emb-a"],
            ollama_reachable=False,
        )

    assert _select_kwargs(mock_st, "model_selector")["disabled"] is True
    assert _select_kwargs(mock_st, "embedding_model_selector")["disabled"] is True
