"""Phase 3: smart-scroller JS contract guards (no browser required).

- Self-heal yank guard uses the single ``BOTTOM_STICK_PX`` constant — the
  old hardcoded ``4px`` vs wheel ``8px`` mismatch latched auto-follow off.
- Initial entry yanks to the bottom exactly once; later reruns respect the
  latch instead of dragging readers of past turns to the bottom.
- ``inject_custom_css`` always emits the deterministic probe slots, even
  when the CSS file cannot be loaded (empty-CSS fallback, file log only —
  ``st.error`` would add a DOM node and revive the upload-collapse).
"""

from unittest.mock import patch

import ui.ui as ui_module
from ui.components.autoscroll_stick import BOTTOM_STICK_PX


def _js_source() -> str:
    import inspect

    return inspect.getsource(ui_module.inject_custom_css)


def test_yank_guard_uses_single_stick_constant() -> None:
    """No magic 4px in the yank guard; BOTTOM_STICK_PX is interpolated."""
    src = _js_source()
    assert "yank - container.clientHeight - {BOTTOM_STICK_PX}" in src
    assert "clientHeight - 4" not in src
    assert BOTTOM_STICK_PX == 8


def test_initial_yank_fires_once_then_respects_latch() -> None:
    """First observer setup forces bottom; subsequent setups do not."""
    src = _js_source()
    assert "initialYanked" in src
    assert "scrollState.initialYanked = true" in src
    # Explicit user intent (Enter / send button) keeps force.
    assert "scrollToBottom(true)" in src
    # Rerun path falls back to latch-respecting mode.
    assert "scrollToBottom(false)" in src


def test_css_failure_keeps_deterministic_probe_slots() -> None:
    """CSS load failure falls back to empty CSS and still emits probes."""
    with (
        patch.object(ui_module, "_load_css", side_effect=OSError("gone")),
        patch.object(ui_module.st, "iframe") as mock_iframe,
    ):
        ui_module.inject_custom_css()
    # Header probe + CSS head-injection probe: slot count stays deterministic.
    assert mock_iframe.call_count == 2
