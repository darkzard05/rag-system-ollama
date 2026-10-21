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
        """PREPARING 상태에서 게이지는 숨기되, 스텝 안내(stepper)와 캡션이 부드럽게 제공되는지 검증."""
        build = initial_build_state("sample_document.pdf", "hash_abc_999")
        build.phase = BuildPhase.PREPARING

        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "running"
        assert vm.expanded is True

        # 1. 0% 막대 경쟁 방지: progress는 None 유지
        assert vm.progress is None

        # 2. 자연스러운 단계 인지를 위한 stepper는 준비 상태(○ 대기)로 출력되어야 함
        assert vm.stepper is not None
        assert "○" in vm.stepper
        assert "추출" in vm.stepper

        # 3. 빈 공간 방지를 위한 준비 안내 캡션이 존재해야 함
        assert vm.caption is not None
        assert vm.caption == t("build_analyzing")
        assert "sample_document.pdf" in vm.label


if __name__ == "__main__":
    unittest.main()
