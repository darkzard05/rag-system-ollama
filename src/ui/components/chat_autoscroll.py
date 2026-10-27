"""Chat auto-scroll sentinel (observe-only bottom-proximity probe).

Emits a zero-height ``st.iframe`` probe at the end of the chat timeline. The
``<script>``-only payload plants the ``#chat-autoscroll-sentinel`` marker and
holds an ``IntersectionObserver`` on it, recording edge visibility in the
``window.parent.__chatSentinel`` singleton (with observer disconnect guard).

The probe deliberately never scrolls. Auto-follow has exactly one writer — the
ui.py smart scroller (``window.parent.__chatScrollState``), which this module
never reads or assigns.
"""

from __future__ import annotations

from typing import cast

import streamlit as st

BOTTOM_STICK_PX: int = 8


def stick_math(
    scroll_height: int,
    scroll_top: int,
    client_height: int,
    threshold: int = BOTTOM_STICK_PX,
) -> dict[str, bool | int]:
    """Compute distance to bottom and whether stick-to-bottom is active."""
    if scroll_height <= 0 or client_height <= 0:
        return {"dist": -1, "stick_active": False}
    dist: int = scroll_height - scroll_top - client_height
    return {"dist": dist, "stick_active": dist <= threshold}


def edge_latch(
    user_scrolled_up: bool,
    scroll_delta_y: int,
    dist_to_bottom: int,
    threshold: int = BOTTOM_STICK_PX,
) -> bool:
    """Return True when the user scroll latch stays engaged."""
    return user_scrolled_up and (dist_to_bottom > threshold or scroll_delta_y != 0)


def flag_read(
    is_generating_answer: bool, active_stream_msg_id: str, msg_id: str
) -> bool:
    """Return True only while the given message is the active stream."""
    return (
        bool(is_generating_answer)
        and bool(active_stream_msg_id)
        and active_stream_msg_id == msg_id
    )


SENTINEL_ID: str = "chat-autoscroll-sentinel"
CHAT_AUTOSCROLL_IFRAME_KEY: str = "chat_autoscroll_sentinel_v1"

__all__ = [
    "BOTTOM_STICK_PX",
    "CHAT_AUTOSCROLL_IFRAME_KEY",
    "SENTINEL_ID",
    "build_autoscroll_js",
    "edge_latch",
    "flag_read",
    "render_autoscroll_sentinel",
    "stick_math",
]


def build_autoscroll_js(force: bool) -> str:
    """Return the ``<script>``-only sentinel probe payload for ``st.iframe``."""
    force_literal = "true" if force else "false"
    js = """
        <script>
        try {
            const doc = window.parent.document;
            const FORCE = __FORCE__;
            void FORCE;
            const SENTINEL_ID = "__SENTINEL_ID__";
            const SCROLLER_SEL = '[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]';
            const getScroller = () => {
                const input = doc.querySelector('[data-testid="stChatInput"]');
                if (input) {
                    const col = input.closest('[data-testid="stColumn"]');
                    const vb = col && col.querySelector(':scope > [data-testid="stVerticalBlock"]');
                    if (vb) return vb;
                }
                return doc.querySelector(SCROLLER_SEL);
            };
            const prev = window.parent.__chatSentinel;
            if (prev && prev.observer) {
                prev.observer.disconnect();
            }
            if (!window.parent.__chatSentinel) {
                window.parent.__chatSentinel = {
                    observer: null,
                    wasVisible: false
                };
            }
            const sentinelState = window.parent.__chatSentinel;
            sentinelState.observer = null;

            let marker = doc.getElementById(SENTINEL_ID);
            if (!marker) {
                marker = doc.createElement("div");
                marker.id = SENTINEL_ID;
                const scroller = getScroller();
                const inputWrap = scroller ? scroller.querySelector(
                    ':scope > [data-testid="stElementContainer"]:has([data-testid="stChatInput"]),'
                    + ' :scope > [data-testid="stLayoutWrapper"]:has([data-testid="stChatInput"])'
                ) : null;
                if (scroller) {
                    if (inputWrap) scroller.insertBefore(marker, inputWrap);
                    else scroller.appendChild(marker);
                }
            }

            const scroller = getScroller();
            if (scroller && marker) {
                sentinelState.observer = new IntersectionObserver((entries) => {
                    for (const entry of entries) {
                        sentinelState.wasVisible = entry.isIntersecting === true;
                    }
                }, { root: scroller, threshold: 0.1 });
                sentinelState.observer.observe(marker);
            }
        } catch (e) {}
        </script>
        """
    return js.replace("__FORCE__", force_literal).replace(
        "__SENTINEL_ID__", SENTINEL_ID
    )


def render_autoscroll_sentinel(session_id: str, force: bool = False) -> None:
    """Emit the auto-scroll sentinel probe after the chat timeline."""
    _ = session_id
    _ = CHAT_AUTOSCROLL_IFRAME_KEY
    payload = build_autoscroll_js(force)
    _ = cast("object", st.iframe(payload, height="content"))
