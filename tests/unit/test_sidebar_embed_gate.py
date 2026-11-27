"""Embedding selector follows the same Ollama reachability gate as the LLM one."""

from __future__ import annotations

from unittest.mock import patch

import ui.components.sidebar as sidebar_mod


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
        mock_st.session_state.get.return_value = 0
        mock_st.columns.side_effect = lambda spec, *a, **k: (mock_st,) * len(spec)
        sidebar_mod._render_settings_internal(
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: None,
            False,
            False,
            None,
            ["llm-a", "emb-a"],
            ollama_reachable=False,
        )

    assert _select_kwargs(mock_st, "model_selector_d0")["disabled"] is True
    assert _select_kwargs(mock_st, "embedding_model_selector_d0")["disabled"] is True
