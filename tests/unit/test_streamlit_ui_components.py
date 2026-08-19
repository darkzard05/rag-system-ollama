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

from common.config import MSG_CHAT_GUIDE  # noqa: E402

_SMOKE_SCRIPT = str(Path(__file__).parent / "_streamlit_ui_smoke.py")


class TestStreamlitUIComponents(unittest.TestCase):
    def test_initial_state_elements(self):
        """Startup UI elements render: brand, guide message, model selectbox."""
        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()

        # Sidebar brand (sidebar.py: _render_sidebar_logo)
        assert any("GraphRAG-Ollama" in str(m.value) for m in at.sidebar.markdown), (
            "Sidebar brand 'GraphRAG-Ollama' not rendered"
        )

        # Chat guide message. NOTE: the string is config-driven
        # (common.config.MSG_CHAT_GUIDE, sourced from config.yml), so we assert
        # against the imported value rather than a hardcoded literal — the
        # previous integration test hard-coded a Korean string that no longer
        # matched the English config value.
        assert any(
            MSG_CHAT_GUIDE in str(m.value) for m in at.chat_message[0].markdown
        ), f"Chat guide message '{MSG_CHAT_GUIDE}' not rendered"

        # Model + embedding selectboxes (sidebar.py: render_settings_content)
        assert len(at.sidebar.selectbox) >= 1, "No model selectbox rendered"

    def test_chat_input_renders(self):
        """Chat input area renders without a full app boot."""
        at = AppTest.from_file(_SMOKE_SCRIPT, default_timeout=60).run()

        assert len(at.chat_input) >= 1, "Chat input not rendered"


if __name__ == "__main__":
    unittest.main()
