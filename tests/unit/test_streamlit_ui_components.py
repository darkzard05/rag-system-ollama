"""Unit tests for static UI component rendering.

Converted from ``tests/integration/test_streamlit_app.py`` (integration) to a
lightweight unit test after Momus review (verdict A). The original booted the
entire ``src/main.py`` (global ``AsyncWorker`` loop, background worker submit,
``SessionManager.set_ui_sync``, Windows integrity check) which caused
intermittent incomplete renders under AppTest. UI components are independently
renderable pure functions, so we drive a minimal render script
(``_streamlit_ui_smoke.py``) via ``AppTest.from_file`` — no app boot, no real
Ollama models (``IS_CI_TEST=true`` -> ``core.model_loader`` stubs).

``from_function`` is intentionally avoided: it fails in this environment
(subprocess ``import streamlit`` error: ``st is not defined``).
"""

import os
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

os.environ.setdefault("IS_CI_TEST", "true")

_SMOKE_SCRIPT = str(Path(__file__).parent / "_streamlit_ui_smoke.py")


class TestStreamlitUIComponents(unittest.TestCase):
    def test_initial_state_elements(self):
        """Startup UI elements render: brand, onboarding guide, model selectbox."""
        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()

        # Sidebar brand (sidebar.py: _render_sidebar_logo)
        assert any("GraphRAG-Ollama" in str(m.value) for m in at.sidebar.markdown), (
            "Sidebar brand 'GraphRAG-Ollama' not rendered"
        )

        # Onboarding guide panel. NOTE: the initial chat area no longer shows
        # the config-driven MSG_CHAT_GUIDE string — with no messages and no
        # uploaded file, _render_guidance_panel() renders a Korean step-by-step
        # onboarding card instead, so we assert against that content.
        assert any(
            "업로드하신 PDF" in str(m.value) for m in at.chat_message[0].markdown
        ), "Onboarding guide panel not rendered"

        # Advanced settings moved from the sidebar into a centered modal.
        assert len(at.sidebar.selectbox) == 0, "Settings still rendered in the sidebar"
        buttons = [b for b in at.sidebar.button if b.key == "settings_dialog_button"]
        assert len(buttons) == 1, "Settings button not rendered in the sidebar"
        buttons[0].click().run()
        assert len(at.selectbox) == 4, "Two-column settings did not open"
        assert len(at.toggle) == 3, "Runtime setting toggles are missing"

    @staticmethod
    def _open_settings(at):
        buttons = [b for b in at.sidebar.button if b.key == "settings_dialog_button"]
        assert len(buttons) == 1, "Settings button not rendered in the sidebar"
        buttons[0].click().run()

    @staticmethod
    def _toggle(at, base_key):
        return [t for t in at.toggle if t.key and t.key.startswith(base_key)]

    def test_settings_save_commits_runtime_value(self):
        """A staged toggle reaches core only when Save is pressed."""
        from common.config import ENABLE_VECTOR_CACHE

        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()
        self._open_settings(at)
        toggles = self._toggle(at, "setting_vector_cache")
        assert len(toggles) == 1, "Vector cache toggle not rendered"
        try:
            toggles[0].set_value(False).run()
            assert bool(ENABLE_VECTOR_CACHE) is True, "change applied before Save"
            save = [b for b in at.button if b.key == "settings_save_btn"]
            assert len(save) == 1, "Save button not rendered"
            save[0].click().run()
            assert bool(ENABLE_VECTOR_CACHE) is False
            assert not [b for b in at.button if b.key == "settings_save_btn"]
        finally:
            ENABLE_VECTOR_CACHE.reset()

    def test_settings_cancel_discards_runtime_value(self):
        """Cancel closes the dialog without applying staged changes."""
        from common.config import ENABLE_VECTOR_CACHE

        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()
        self._open_settings(at)
        try:
            self._toggle(at, "setting_vector_cache")[0].set_value(False).run()
            cancel = [b for b in at.button if b.key == "settings_cancel_btn"]
            assert len(cancel) == 1, "Cancel button not rendered"
            cancel[0].click().run()
            assert bool(ENABLE_VECTOR_CACHE) is True
            assert not [b for b in at.button if b.key == "settings_cancel_btn"]
            # Reopening remounts the widgets with the live value, not the draft.
            self._open_settings(at)
            assert bool(self._toggle(at, "setting_vector_cache")[0].value) is True
        finally:
            ENABLE_VECTOR_CACHE.reset()

    def test_settings_cancel_keeps_live_language(self):
        """Language is staged too: Cancel must not switch the active locale."""
        from ui.strings import LANG

        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()
        self._open_settings(at)
        original = LANG
        selector = [
            s for s in at.selectbox if s.key and s.key.startswith("language_selector")
        ]
        assert len(selector) == 1
        other = "English" if original == "ko" else "한국어"
        selector[0].set_value(other).run()
        cancel = [b for b in at.button if b.key == "settings_cancel_btn"][0]
        cancel.click().run()
        assert LANG == original

    def test_settings_reset_defers_confirm_dialog(self):
        """Reset inside the settings modal must not nest two dialogs."""
        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()
        settings = [b for b in at.sidebar.button if b.key == "settings_dialog_button"]
        settings[0].click().run()
        reset = [b for b in at.button if b.key and b.key.startswith("reset_all_")]
        assert len(reset) == 1, "Reset All not rendered in the settings modal"
        reset[0].click().run()
        assert not at.exception, at.exception
        assert any(b.key == "reset_confirm_cancel_btn" for b in at.button), (
            "Reset confirmation did not open after the settings modal closed"
        )

    def test_chat_input_renders(self):
        """Chat input area renders without a full app boot."""
        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()

        assert len(at.chat_input) >= 1, "Chat input not rendered"


if __name__ == "__main__":
    unittest.main()
