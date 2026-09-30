"""Chat auto-scroll sentinel (observe-only bottom-proximity probe).

Emits a zero-height ``st.iframe`` probe at the end of the chat timeline. The
``<script>``-only payload plants the ``#chat-autoscroll-sentinel`` marker and
holds an ``IntersectionObserver`` on it, recording edge visibility in the
``window.parent.__chatSentinel`` singleton (with observer disconnect guard).

The probe deliberately never scrolls. Auto-follow has exactly one writer — the
ui.py smart scroller (``window.parent.__chatScrollState``), which this module
never reads or assigns. An earlier revision also scrolled here, which made it a
second writer racing the smart scroller: ``force`` re-fired a scroll on every
streaming rerun, the browser coalesced it until it dispatched against a grown
``scrollHeight``, and the smart scroller's self-heal read that phantom distance
as user intent and latched auto-follow off mid-stream.
"""

from __future__ import annotations

from typing import cast

import streamlit as st

from ui.components.autoscroll_edge import edge_latch
from ui.components.autoscroll_flag import flag_read
from ui.components.autoscroll_stick import stick_math

SENTINEL_ID: str = "chat-autoscroll-sentinel"
CHAT_AUTOSCROLL_IFRAME_KEY: str = "chat_autoscroll_sentinel_v1"

__all__ = [
    "CHAT_AUTOSCROLL_IFRAME_KEY",
    "SENTINEL_ID",
    "build_autoscroll_js",
    "edge_latch",
    "flag_read",
    "render_autoscroll_sentinel",
    "stick_math",
]


def build_autoscroll_js(force: bool) -> str:
    """Return the ``<script>``-only sentinel probe payload for ``st.iframe``.

    The probe is observe-only: it plants ``#chat-autoscroll-sentinel`` and keeps
    an ``IntersectionObserver`` liveness handle on it. It never scrolls.

    ``force`` is still templated in for call-site compatibility but no longer
    has an effect — scrolling during a stream is the ui.py smart scroller's job
    (see :func:`render_autoscroll_sentinel`).
    """
    force_literal = "true" if force else "false"
    js = """
        <script>
        try {
            const doc = window.parent.document;
            // Intentionally unused: the probe does not scroll. Retained so the
            // __FORCE__ substitution stays valid for existing callers.
            const FORCE = __FORCE__;
            void FORCE;
            const SENTINEL_ID = "__SENTINEL_ID__";
            const SCROLLER_SEL = '[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]';
            // Anchor on the chat input, NOT `:last-child`. With a real PDF
            // loaded the main block holds 5 stColumn nodes (gutter/spacer
            // columns included) and `:last-child` lands on a 211x40 sliver
            // whose scrollHeight === clientHeight, so every scroll is a
            // permanent no-op. stChatInput is always present in the chat
            // column and lives inside the scroller itself.
            const getScroller = () => {
                const input = doc.querySelector('[data-testid="stChatInput"]');
                if (input) {
                    const col = input.closest('[data-testid="stColumn"]');
                    const vb = col && col.querySelector(':scope > [data-testid="stVerticalBlock"]');
                    if (vb) return vb;
                }
                return doc.querySelector(SCROLLER_SEL);
            };
            // Singleton with disconnect guard. __chatScrollState is owned by
            // the ui.py smart scroller — never read or assign it here.
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

            // Ensure the trailing marker exists as the last message-flow child
            // (before the sticky input wrapper).
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

            // PROBE ONLY — this observer never scrolls. The marker exists so
            // bottom proximity is observable, but auto-follow has exactly one
            // writer: the ui.py smart scroller. A second writer here raced it —
            // `force` re-fired scrollNow() on every streaming rerun, and the
            // browser coalesced that scroll until it dispatched against a grown
            // scrollHeight, so the ui.py self-heal read a phantom dist and
            // latched auto-follow off mid-stream (A4 dist=53).
            const scroller = getScroller();
            if (scroller && marker) {
                sentinelState.observer = new IntersectionObserver((entries) => {
                    for (const entry of entries) {
                        sentinelState.wasVisible = entry.isIntersecting === true;
                    }
                }, { root: scroller, threshold: 0 });
                sentinelState.observer.observe(marker);
            }
        } catch (e) {}
        </script>
        """
    return js.replace("__FORCE__", force_literal).replace(
        "__SENTINEL_ID__", SENTINEL_ID
    )


def render_autoscroll_sentinel(session_id: str, force: bool = False) -> None:
    """Emit the auto-scroll sentinel probe after the chat timeline.

    ``session_id`` scopes the call to the current session (the iframe slot
    itself uses the constant :data:`CHAT_AUTOSCROLL_IFRAME_KEY` so reruns
    replace instead of stacking). ``force`` jumps straight to the bottom
    while a stream is active.
    """
    _ = session_id
    payload = build_autoscroll_js(force)
    try:
        # Constant key => rerun-replace in the deterministic post-timeline
        # slot (1.60 types st.iframe without key; fallback below covers it).
        _ = cast(
            "object",
            st.iframe(
                payload,
                height="content",
                key=CHAT_AUTOSCROLL_IFRAME_KEY,  # pyright: ignore[reportCallIssue]
            ),
        )
    except TypeError:
        # Streamlit 1.60 st.iframe has no key param; the deterministic
        # emission slot (always right after the timeline) still replaces
        # the probe on rerun.
        _ = cast("object", st.iframe(payload, height="content"))
