"""Repro for defect 1 (strategy C+D): focus-order mitigation with zero visual space.

The DOM order (input before messages, see ``ui.render_main_content``) is kept
for streaming stability, so the order notice reaches AT users without taking
visual space:

- C: the existing ``inject_custom_css`` JS probe appends the localized notice
  to the chat textarea's accessible name (same f-string pattern as
  ``css_json``), re-applied on every probe run + DOM mutation (self-healing).
- D: the empty-state onboarding panel repeats the notice (first-visit only).
- No permanent caption/element may live in ``render_chat_input_area``.
"""

from __future__ import annotations

import inspect

import ui.components.chat as chat_module
import ui.components.chat_build as chat_build_module
import ui.ui as ui_module
from ui.strings import UI_STRINGS


def test_focus_order_has_zero_space_mitigation() -> None:
    """Zero visual space + AT notice via JS aria wiring + onboarding line."""
    input_src = inspect.getsource(chat_module.render_chat_input_area)
    assert "focus_order_notice" not in input_src, (
        "render_chat_input_area must not render a permanent notice element "
        "(zero visual space; the notice lives in the textarea aria-label)"
    )

    js_src = inspect.getsource(ui_module.inject_custom_css)
    assert "applyA11yLabel" in js_src, "probe must define the aria-label applier"
    assert "aria-label" in js_src, "probe must write the textarea accessible name"
    assert 't("focus_order_notice")' in js_src, (
        "probe must inject the localized notice (same pattern as css_json)"
    )

    guidance_src = inspect.getsource(chat_build_module._render_guidance_panel)
    assert "focus_order_notice" in guidance_src, (
        "empty-state onboarding must repeat the order notice (first-visit only)"
    )

    assert "focus_order_notice" in UI_STRINGS["en"], "en table needs the key"
    assert "focus_order_notice" in UI_STRINGS["ko"], "ko table needs the key"
    assert UI_STRINGS["en"]["focus_order_notice"].strip()
    assert UI_STRINGS["ko"]["focus_order_notice"].strip()
