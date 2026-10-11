"""UX-8: core-concept inline helps on uploader/LLM/embedding widgets."""

from __future__ import annotations

import inspect

import ui.components.sidebar as sidebar_mod
from ui.strings import UI_STRINGS


def test_help_keys_complete_en_ko() -> None:
    for key in (
        "sidebar_upload_help",
        "sidebar_llm_help",
        "sidebar_embedding_help",
    ):
        assert UI_STRINGS["en"][key].strip()
        assert UI_STRINGS["ko"][key].strip()


def test_sidebar_widgets_wire_help() -> None:
    source = inspect.getsource(sidebar_mod._render_settings_internal)
    assert 'help=t("sidebar_upload_help")' in source
    assert 'help=t("sidebar_llm_help")' in source
    assert 'help=t("sidebar_embedding_help")' in source
