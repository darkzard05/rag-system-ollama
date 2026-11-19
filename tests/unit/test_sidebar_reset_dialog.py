"""Tests for sidebar reset confirmation dialog (_confirm_reset_all) and rerun/reset behaviors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ui.components.sidebar as sidebar_mod
from core.session import SessionManager


def test_confirm_reset_all_neither() -> None:
    """정상 케이스 1: 아무 버튼도 클릭하지 않음 -> SessionManager.reset_all_state 및 st.rerun 미호출."""
    func = getattr(
        sidebar_mod._confirm_reset_all, "__wrapped__", sidebar_mod._confirm_reset_all
    )
    with (
        patch("ui.components.sidebar.st") as mock_st,
        patch.object(SessionManager, "reset_all_state") as mock_reset,
    ):
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_st.button.return_value = False
        func()
        mock_reset.assert_not_called()
        mock_st.rerun.assert_not_called()


def test_confirm_reset_all_cancel() -> None:
    """버그 재현 케이스 1: 취소 버튼 클릭 -> SessionManager.reset_all_state 미호출, st.rerun() 1회 호출."""
    func = getattr(
        sidebar_mod._confirm_reset_all, "__wrapped__", sidebar_mod._confirm_reset_all
    )
    with (
        patch("ui.components.sidebar.st") as mock_st,
        patch.object(SessionManager, "reset_all_state") as mock_reset,
    ):
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_st.button.side_effect = (
            lambda label, **kwargs: kwargs.get("key") == "reset_confirm_cancel_btn"
        )
        func()
        mock_reset.assert_not_called()
        mock_st.rerun.assert_called_once()


def test_confirm_reset_all_confirm() -> None:
    """버그 재현 케이스 2: 초기화 실행 버튼 클릭 -> SessionManager.reset_all_state 1회 호출 후 st.rerun() 1회 호출."""
    func = getattr(
        sidebar_mod._confirm_reset_all, "__wrapped__", sidebar_mod._confirm_reset_all
    )
    with (
        patch("ui.components.sidebar.st") as mock_st,
        patch.object(SessionManager, "reset_all_state") as mock_reset,
    ):
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_st.button.side_effect = (
            lambda label, **kwargs: kwargs.get("key") == "reset_confirm_btn"
        )
        func()
        mock_reset.assert_called_once()
        mock_st.rerun.assert_called_once()
