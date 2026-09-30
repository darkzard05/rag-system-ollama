"""Deterministic auto-scroll harness (zero Ollama, zero models, zero PDFs).

Drives the REAL production auto-scroll stack against a REAL
Streamlit-rendered DOM:

* ``ui.ui.inject_custom_css`` — the real ``main.css`` plus the real header
  probe and the real ``MutationObserver`` smart-scroller JS (``st.iframe``).
* ``ui.components.chat_autoscroll.render_autoscroll_sentinel`` — the real
  ``IntersectionObserver`` sentinel probe (``build_autoscroll_js``).

Message growth is produced with ``st.write_stream`` fed by a fake generator
that yields canned chunks with small delays, so the browser receives
progressive deltas exactly like a live LLM stream, without any backend.

Layout mirrors production (``st.columns([42, 58])``; chat input rendered
BEFORE the messages per the TASK5-a verdict; sentinel emitted AFTER the
timeline per ``render_chat_messages_area``).

A Playwright test clicks the sidebar ``append_stream`` button, which triggers
one rerun whose ``st.write_stream`` call streams ~60 chunks over ~2s. The
test then observes whether the chat scroller followed.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Generator

import streamlit as st

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ui.components.chat_autoscroll import render_autoscroll_sentinel  # noqa: E402
from ui.ui import inject_custom_css  # noqa: E402

SEED_MESSAGE_COUNT = 30
STREAM_CHUNK_COUNT = 60
STREAM_CHUNK_DELAY_S = 0.03
STATUS_REWRITE_EVERY = 10
AUX_SWAP_AT_CHUNK = 30
HARNESS_SESSION_ID = "autoscroll-harness"

SEED_TEXT = (
    "Seed message for auto-scroll observation. "
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "This paragraph exists only to force the chat column into overflow. "
)


def _fake_token_stream(n: int) -> Generator[str, None, None]:
    """Yield ``n`` canned chunks with a small delay, like an LLM stream.

    Mirrors the production ``_content_generator`` side-channel pattern: every
    ``STATUS_REWRITE_EVERY`` chunks the status caption ABOVE the stream text
    is rewritten, and mid-stream the aux slot is emptied and swapped for an
    expander — i.e. node replacement above the fold while text grows below.
    """
    aux_slot = st.session_state.get("_harness_aux_slot")
    status_slot = st.session_state.get("_harness_status_slot")
    for i in range(n):
        if status_slot is not None and i % STATUS_REWRITE_EVERY == 0:
            status_slot.caption(f"Retrieving evidence… phase {i // 10}/6 ▍")
        if aux_slot is not None and i == AUX_SWAP_AT_CHUNK:
            aux_slot.empty()
            aux_expander = aux_slot.expander("Answer details (harness)", expanded=False)
            aux_expander.markdown("Canned thought trace for scroll observation.")
        yield f"token-{i:03d} canned streaming chunk for auto-scroll check. "
        time.sleep(STREAM_CHUNK_DELAY_S)


def _seed_messages() -> list[dict[str, str]]:
    """Build the initial overflowing history (alternating roles)."""
    seeds: list[dict[str, str]] = []
    for i in range(SEED_MESSAGE_COUNT):
        role = "user" if i % 2 == 0 else "assistant"
        seeds.append({"role": role, "text": f"Seed {i:02d}. {SEED_TEXT * 3}"})
    return seeds


st.set_page_config(page_title="autoscroll-harness", layout="wide")

# REAL production CSS + REAL production scroller JS (header probe,
# MutationObserver smart scroller). No Ollama, no RAG involved.
inject_custom_css()

if "msgs" not in st.session_state:
    st.session_state.msgs = _seed_messages()

append_clicked: bool = bool(st.sidebar.button("append_stream", key="append_stream_btn"))
st.sidebar.caption("harness: no Ollama, no PDF, canned st.write_stream only")

col_pdf, col_chat = st.columns([42, 58], gap="small")

with col_pdf:
    st.markdown("## PDF placeholder")
    st.markdown("Left column stands in for the PDF viewer.")
    for i in range(6):
        st.markdown(f"Placeholder paragraph {i}. " + SEED_TEXT)

with col_chat:
    # Production order (TASK5-a): input widget exists BEFORE the streaming
    # loop starts; visual bottom-pin comes from main.css (order:1 + sticky).
    _ = st.chat_input("Ask a question (harness input; submits nothing)")

    for msg in st.session_state.msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    if append_clicked:
        # Production structure (chat.py::_render_streaming_with_write_stream):
        # aux slot + status slot ABOVE the stream text inside one chat_message;
        # status rewritten per update, aux swapped mid-stream, both emptied in
        # the finally-equivalent teardown after write_stream returns.
        with st.chat_message("assistant"):
            aux_ph = st.empty()
            status_ph = st.empty()
            status_ph.caption("Analyzing the question… ▍")
            st.session_state["_harness_aux_slot"] = aux_ph
            st.session_state["_harness_status_slot"] = status_ph
            try:
                streamed = st.write_stream(_fake_token_stream(STREAM_CHUNK_COUNT))
            finally:
                status_ph.empty()
                aux_ph.empty()
                final_expander = aux_ph.expander(
                    "Answer details (final)", expanded=False
                )
                final_expander.markdown("Final canned metadata.")
            st.session_state["_harness_aux_slot"] = None
            st.session_state["_harness_status_slot"] = None
        text = streamed if isinstance(streamed, str) else "".join(streamed)
        st.session_state.msgs.append({"role": "assistant", "text": text})

    # REAL production sentinel probe, emitted after the timeline exactly
    # like render_chat_messages_area does. force=False: quiescent state.
    render_autoscroll_sentinel(HARNESS_SESSION_ID, force=False)
