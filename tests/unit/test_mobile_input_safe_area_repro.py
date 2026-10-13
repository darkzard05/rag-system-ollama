"""Repro for defect 2: mobile fixed chat input occludes content/safe-area.

- test_mobile_fixed_input_respects_safe_area: mobile stChatInput rule must
  reference env(safe-area-inset-bottom) so the iPhone home indicator /
  gesture bar does not cover the input (MDN env() safe-area pattern).
- test_chat_column_reserves_input_space: the mobile chat column must reserve
  bottom space (padding-bottom / scroll-padding-bottom) so the last message
  is not hidden behind the fixed input.
"""

from __future__ import annotations

import pathlib

import ui.components.viewer as viewer_module

_VIEWER_FILE = viewer_module.__file__
assert _VIEWER_FILE is not None
_SRC_ROOT = pathlib.Path(_VIEWER_FILE).parent.parent.parent
_CSS_PATH = _SRC_ROOT / "ui" / "styles" / "main.css"


def _css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def _mobile_block() -> str:
    css = _css()
    return css[css.index("@media (max-width: 768px)") :]


def test_mobile_fixed_input_respects_safe_area() -> None:
    """Fixed mobile input must offset with safe-area-inset-bottom."""
    mobile = _mobile_block()
    assert "safe-area-inset-bottom" in mobile, (
        "mobile fixed stChatInput must use env(safe-area-inset-bottom)"
    )


def test_chat_column_reserves_input_space() -> None:
    """Mobile chat column must reserve space for the fixed input."""
    mobile = _mobile_block()
    anchor = '[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child'
    assert anchor in mobile, "mobile chat-column selector must exist"
    # Scope to the chat-column rule block (up to its closing brace) so the
    # generic/PDF-column padding-bottom rules cannot satisfy this vacuously.
    start = mobile.index(anchor)
    end = mobile.index("}", start)
    block = mobile[start:end]
    assert ("padding-bottom" in block) or ("scroll-padding-bottom" in block), (
        "mobile chat column rule must reserve bottom space for the fixed input"
    )
