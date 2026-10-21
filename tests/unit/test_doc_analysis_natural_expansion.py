import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.build_state import BuildPhase, initial_build_state
from ui.components.doc_analysis.viewmodel import to_viewmodel
from ui.strings import set_lang, t


class TestDocAnalysisNaturalExpansion(unittest.TestCase):
    def setUp(self) -> None:
        set_lang("ko")

    def tearDown(self) -> None:
        set_lang("en")

    def test_preparing_phase_remains_compact_for_viewer_precedence(self):
        """PREPARING 위상: 0% 프로그레스바를 숨기되 패널은 펼쳐져(expanded=True) 스텝퍼와 안내 캡션을 노출함을 검증."""
        build = initial_build_state("report_2026.pdf", "hash_999")
        build.phase = BuildPhase.PREPARING

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        # 1. 패널은 펼쳐진 상태 유지 (expanded=True)
        assert vm.expanded is True, "PREPARING 단계에서도 패널은 expanded=True여야 함"
        # 2. 0% 게이지는 숨김
        assert vm.progress is None
        # 3. 대기 스텝퍼 및 준비 안내 캡션 제공
        assert vm.stepper is not None
        assert "○" in vm.stepper
        assert vm.caption == t("build_analyzing")
        assert "report_2026.pdf" in vm.label

    def test_extracting_phase_naturally_expands_with_full_details(self):
        """EXTRACTING 위상: 실제 텍스트 분석에 진입하면 expanded=True로 확장되며 스텝퍼와 진행바가 전개됨을 검증."""
        build = initial_build_state("report_2026.pdf", "hash_999")
        build.phase = BuildPhase.EXTRACTING
        build.pct = 25

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        # 분석 시작 시 expanded=True로 자연스럽게 확장 전개
        assert vm.expanded is True, (
            "EXTRACTING 단계부터는 패널이 expanded=True로 확장되어야 함"
        )
        assert vm.progress == 0.25
        assert vm.stepper is not None
        assert "추출" in vm.stepper
        assert "25% 완료" in vm.caption


if __name__ == "__main__":
    unittest.main()
