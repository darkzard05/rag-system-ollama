"""fix-001: PDF 뷰어 fragment의 run_every 폴링 제거 검증.

뷰어 fragment는 수동 트리거(`@st.fragment()` — run_every 없음)로만 동작해야
한다. 프로덕션에서 `pdf_annotations`는 세팅되지 않고, `pdf_target_page`의
유일한 세터가 `st.rerun()`을 동반하므로, 2s 폴링은 필요 없다.
이 테스트는 (1) 데코레이터가 run_every를 다시 도입하지 않도록 잠그고,
(2) 빈 상태에서 뷰어가 결정적 단일 요소로 렌더됨(C8 구조 가드)을 확인한다.
"""

import inspect
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("IS_CI_TEST", "true")

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from common.config import MSG_PDF_VIEWER_NO_FILE  # noqa: E402
from ui.components.viewer import render_pdf_area  # noqa: E402

_VIEWER_SMOKE = str(Path(__file__).parent / "_viewer_smoke.py")


def test_viewer_fragment_has_no_run_every() -> None:
    """run_every 재도입 금지: 데코레이터 소스에서 run_every 부재 확인."""
    source = inspect.getsource(render_pdf_area)
    decorator = source.split("\n", 1)[0].strip()
    assert decorator.startswith("@st.fragment")
    assert "run_every" not in decorator, (
        "run_every 재도입 금지(fix-001): 판별 근거는 뷰어 상태의 모든 세터가 "
        "fragment 콜백 또는 st.rerun() 경로를 사용한다는 분석(계획 §3.2)에 기반함"
    )


class TestViewerFragmentStructure(unittest.TestCase):
    def test_empty_state_renders_single_deterministic_element(self) -> None:
        """C8 가드: 빈 상태에서 st.info 단일 요소만 렌더(구조 결정성)."""
        at = AppTest.from_file(_VIEWER_SMOKE, default_timeout=60).run()
        assert not at.exception, (
            f"빈 상태 렌더에서 예외가 발생하면 안 됨: {at.exception}"
        )
        assert len(at.info) == 1, "빈 상태 뷰어는 안내 st.info 하나만 렌더해야 함"
        assert str(at.info[0].value) == MSG_PDF_VIEWER_NO_FILE
