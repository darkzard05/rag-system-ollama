"""Headless UI render script for unit tests (no full app boot).

Driven by ``AppTest.from_file`` from the unit test. Mirrors the layout that
``src/main.py`` (sidebar) and ``src/ui/ui.py`` (chat column) produce, but
deliberately skips the heavy ``main.py`` import side-effects (global
``AsyncWorker`` loop, background worker submit, ``SessionManager.set_ui_sync``,
Windows integrity check). Those are covered by the integration boot tests, not
here. ``IS_CI_TEST`` is set by the test so ``core.model_loader`` returns stubs
and no real Ollama model is ever loaded.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

os.environ.setdefault("IS_CI_TEST", "true")

from common.config import DEFAULT_OLLAMA_MODEL  # noqa: E402
from core.session import SessionManager  # noqa: E402
from ui.components.chat import (  # noqa: E402
    render_chat_input_area,
    render_chat_messages_area,
)
from ui.components.sidebar import render_settings_content  # noqa: E402

# Seed a minimal session so the chat areas render deterministically.
SessionManager.set_session_id("unit-test-session")
SessionManager.set("is_ready_for_chat", True, session_id="unit-test-session")

import streamlit as st  # noqa: E402

with st.sidebar:
    render_settings_content(
        file_uploader_callback=lambda: None,
        model_selector_callback=lambda: None,
        embedding_selector_callback=lambda: None,
        new_chat_callback=lambda: None,
        refresh_models_callback=lambda: None,
        is_generating=False,
        is_swapping_model=False,
        current_file_name=None,
        available_models=[DEFAULT_OLLAMA_MODEL],
    )

col_pdf, col_chat = st.columns([1, 2], gap="small")
with col_pdf:
    st.write("pdf-area-placeholder")
with col_chat:
    render_chat_messages_area()
    render_chat_input_area()
