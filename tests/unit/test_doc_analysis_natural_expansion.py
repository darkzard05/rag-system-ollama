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
        """PREPARING 위상: 뷰어 1페이지 선행성을 위해 패널이 컴팩트(expanded=False)하게 유지되는지 검증."""
        build = initial_build_state("report_2026.pdf", "hash_999")
        build.phase = BuildPhase.PREPARING

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        # 1. 뷰어 선행 렌더링을 위해 expanded=False로 컴팩트하게 대기
        assert vm.expanded is False, "PREPARING 단계에서는 패널이 expanded=False여야 함"
        # 2. 게이지와 장황한 스텝퍼는 생략
        assert vm.progress is None
        assert vm.stepper is None
        # 3. 간결한 준비 안내 캡션 제공
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
