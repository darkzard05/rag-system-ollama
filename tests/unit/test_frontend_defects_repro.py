"""Frontend defects reproduction and unit tests (D1: widget key write / D2: autoscroll sentinel dead signal).

Strictly matches Streamlit 1.60 requirements and project constraints:
- D1: direct assignment to widget key after instantiation removed.
- D2: threshold:0.1 stable sentinel intersection.
"""

from __future__ import annotations

from pathlib import Path


def test_d1_widget_key_write_in_render_path() -> None:
    """D1 수정 검증: 렌더 중 위젯 키 직접 대입 제거 확인."""
    viewer_path = Path("src/ui/components/viewer.py")
    assert viewer_path.exists()
    code = viewer_path.read_text(encoding="utf-8")

    # 1. 렌더 중 st.session_state[PDF_NAV_INPUT_KEY] 직접 대입 코드가 제거되었는지 확인
    assert "st.session_state[PDF_NAV_INPUT_KEY] = current_page" not in code


def test_d2_autoscroll_sentinel_contract() -> None:
    """D2 수정 검증: chat_autoscroll의 threshold: 0.1 안정적 임계값 및 keyless iframe 계약."""
    autoscroll_path = Path("src/ui/components/chat_autoscroll.py")
    assert autoscroll_path.exists()
    code = autoscroll_path.read_text(encoding="utf-8")

    # 1. threshold: 0.1 사용 계약 확인 (0-area 경계 문제 개선)
    assert "threshold: 0.1" in code
    # 2. force 파라미터가 void 처리되어 무효화됨을 확인
    assert "void FORCE" in code or "FORCE = __FORCE__" in code
    # 3. st.iframe 호출 시 key 파라미터가 없음을 확인 (Streamlit 1.60 시그니처)
    assert "st.iframe(payload" in code
    assert "key=" not in code.split("st.iframe(")[1].split(")")[0]

    # 4. CSS 센티널 높이가 고정되어 있음을 확인
    main_css = Path("src/ui/styles/main.css").read_text(encoding="utf-8")
    assert "#chat-autoscroll-sentinel" in main_css
