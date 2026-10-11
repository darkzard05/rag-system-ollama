"""INT-4: sidebar explains locked controls while an answer generates."""

from __future__ import annotations

import inspect

import ui.components.sidebar as sidebar_mod
from ui.strings import UI_STRINGS


def test_locked_caption_keys_complete_en_ko() -> None:
    assert UI_STRINGS["en"]["sidebar_locked_generating"].strip()
    assert UI_STRINGS["ko"]["sidebar_locked_generating"].strip()


def test_generating_renders_locked_caption() -> None:
    source = inspect.getsource(sidebar_mod.render_settings_content)
    assert "sidebar_locked_generating" in source
    assert "if is_generating:" in source
