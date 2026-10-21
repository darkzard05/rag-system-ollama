"""Document analysis panel transition tests (viewer mount protection).

PREPARING must render a compact wait state so the first PDF page mount is
not competing with a 0% progress gauge. Real extraction phases must still
surface the full progress controls.
"""

import os
import sys
import unittest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from core.build_state import BuildPhase, initial_build_state  # noqa: E402
from ui.components.doc_analysis.viewmodel import to_viewmodel  # noqa: E402
from ui.strings import set_lang  # noqa: E402


class TestDocAnalysisPanelTransition(unittest.TestCase):
    def setUp(self) -> None:
        set_lang("en")

    def tearDown(self) -> None:
        set_lang("en")

    def test_preparing_phase_renders_clean_transition_state(self):
        """PREPARING must not render a distracting progress bar."""
        build = initial_build_state("document.pdf", "hash_123")
        build.phase = BuildPhase.PREPARING

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        # Keep the viewer mount unobstructed: no gauge/stepper/caption yet.
        assert vm.progress is None
        assert vm.stepper is None
        assert vm.caption is None
        assert "document.pdf" in vm.label

    def test_extracting_phase_renders_full_progress_controls(self):
        """EXTRACTING must restore the gauge, stepper, and caption."""
        build = initial_build_state("document.pdf", "hash_123")
        build.phase = BuildPhase.EXTRACTING
        build.pct = 25

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        assert vm.progress == 0.25
        assert vm.stepper is not None
        assert vm.caption is not None
        assert "25% complete" in vm.caption


if __name__ == "__main__":
    unittest.main()
