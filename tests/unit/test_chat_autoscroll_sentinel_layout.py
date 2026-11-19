"""Test chat autoscroll sentinel layout collapse and marker placement fix.

Validates that main.css collapses the chat column sentinel st.iframe and that
build_autoscroll_js appends the marker to scroller via appendChild.
"""

from __future__ import annotations

from pathlib import Path

import ui.components.chat_autoscroll as sentinel_module


def t_css_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "ui"
        / "styles"
        / "main.css"
    )


def test_main_css_contains_chat_sentinel_iframe_collapse() -> None:
    css_content = t_css_path().read_text(encoding="utf-8")
    assert '[data-testid="stColumn"]:last-child' in css_content
    assert (
        'iframe[title="ui.components.chat_autoscroll.render_autoscroll_sentinel"]'
        in css_content
    )


def test_autoscroll_js_uses_append_child() -> None:
    payload = sentinel_module.build_autoscroll_js(False)
    assert "scroller.appendChild(marker)" in payload
    assert "insertBefore" not in payload
