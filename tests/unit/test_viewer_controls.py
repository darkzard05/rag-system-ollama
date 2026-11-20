import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
from ui.components.viewer import render_pdf_controls
from ui.widget_keys import PDF_NAV_INPUT_KEY


def test_render_pdf_controls_prevents_stale_value_overflow():
    """이전 문서의 높은 페이지 번호가 남아있어도 새 문서의 total_pages에 맞춰 자동 보정되는지 검증"""
    # 1. 20페이지짜리 이전 문서를 보던 상태를 모사하여 세션에 15가 잔류
    st.session_state[PDF_NAV_INPUT_KEY] = 15

    # 2. 새 문서가 3페이지짜리이고 1페이지를 렌더링해야 하는 상황
    current_page = 1
    total_pages = 3

    with (
        patch("streamlit.columns") as mock_columns,
        patch("streamlit.number_input") as mock_number_input,
        patch("streamlit.button") as mock_button,
    ):
        # columns mock 구성 (col_prev, col_page, col_next)
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col]

        render_pdf_controls(current_page, total_pages)

        # 3. st.session_state[PDF_NAV_INPUT_KEY]가 15에서 1로 자동 보정되었는지 검증
        assert st.session_state[PDF_NAV_INPUT_KEY] == 1

        # 4. number_input이 max_value=3으로 정상 호출되었는지 검증 (예외 없이 통과)
        mock_number_input.assert_called_once()
        _, kwargs = mock_number_input.call_args
        assert kwargs["min_value"] == 1
        assert kwargs["max_value"] == 3
        assert kwargs["key"] == PDF_NAV_INPUT_KEY
        assert "value" not in kwargs  # value 인자가 전달되지 않아야 함
