"""문서 분석 패널 PREPARING 단계의 시각적 안내 요소 검증 단위 테스트."""

from __future__ import annotations

import os
import sys
import unittest

# src 경로 추가
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from core.build_state import BuildPhase, initial_build_state
from ui.components.doc_analysis.viewmodel import to_viewmodel
from ui.strings import set_lang, t


class TestDocAnalysisPreparingVisual(unittest.TestCase):
    def setUp(self) -> None:
        set_lang("ko")

    def tearDown(self) -> None:
        set_lang("en")

    def test_preparing_phase_renders_stepper_and_guidance_caption(self):
        """PREPARING 상태에서는 패널을 접고 게이지/스텝퍼 없이 안내 캡션만 제공한다."""
        build = initial_build_state("sample_document.pdf", "hash_abc_999")
        build.phase = BuildPhase.PREPARING

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        assert vm.expanded is False
        assert vm.stepper is None

        # 1. 0% 막대 경쟁 방지: progress는 None 유지
        assert vm.progress is None

        # 2. 빈 공간 방지를 위한 준비 안내 캡션이 존재해야 함
        assert vm.caption is not None
        assert vm.caption == t("build_analyzing")
        assert "sample_document.pdf" in vm.label


if __name__ == "__main__":
    unittest.main()
