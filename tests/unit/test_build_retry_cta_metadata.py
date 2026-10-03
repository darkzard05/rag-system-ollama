"""Build-error retry CTA single-source-of-truth (SSoT) tests.

``chat.py``의 중단-답변 dead-end는 버튼 라벨/도움말을
``get_stopped_answer_actions`` 메타데이터 + ``t()`` 에서 읽는다.
빌드-오류 dead-end(``chat_build.py``)도 같은 계약을 따라야 한다:
렌더된 ``st.button`` 라벨은 ``get_build_error_actions()["cta"][0]`` 과
동일해야 하며, 하드코딩 리터럴과 drift 되어서는 안 된다.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import ui.components.chat_build as chat_build_module
from ui.strings import set_lang, t


def _error_store() -> dict[str, Any]:
    """빌드-오류 분기에 진입하는 SessionManager 스냅샷."""
    return {
        "is_building_rag": False,
        "rebuild_cancelled": False,
        "rebuild_done": False,
        "last_uploaded_file_name": "doc.pdf",
        "rebuild_progress": 0,
        "rebuild_status": "",
        "pdf_processing_error": "boom",
        "pdf_processed": False,
        "doc_stats": {},
        "rebuild_start_time": None,
    }


def _render_error_block(fake_st: MagicMock) -> MagicMock:
    """오류 블록을 렌더하고 retry ``st.button`` 목을 반환한다."""
    store = _error_store()

    class _FakeSessionManager:
        @staticmethod
        def get(key: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
            return store.get(key, default)

    with (
        patch.object(chat_build_module, "st", fake_st),
        patch.object(chat_build_module, "SessionManager", _FakeSessionManager),
        patch.object(chat_build_module, "ui_error", MagicMock()),
    ):
        chat_build_module._render_build_progress_block("sid-test")
    assert fake_st.button.called, "빌드 오류 렌더는 retry 버튼 필요"
    return fake_st.button


def test_retry_button_label_comes_from_cta_metadata() -> None:
    """렌더된 버튼 라벨은 ``actions["cta"][0]`` 과 동일해야 한다."""
    fake_st = MagicMock()
    button = _render_error_block(fake_st)
    args, _ = button.call_args
    expected = chat_build_module.get_build_error_actions("boom")["cta"][0]
    assert args[0] == expected


def test_retry_button_label_follows_metadata_not_literal() -> None:
    """메타데이터를 바꾸면 라벨이 따라간다 (하드코딩이면 실패)."""
    fake_st = MagicMock()
    sentinel: dict[str, Any] = {
        "message": "boom",
        "cause": "boom",
        "retryable": True,
        "retry_kind": "rebuild",
        "cta": ["SENTINEL_CTA_XYZ"],
    }
    with patch.object(
        chat_build_module,
        "get_build_error_actions",
        return_value=sentinel,
    ):
        button = _render_error_block(fake_st)
    args, _ = button.call_args
    assert args[0] == "SENTINEL_CTA_XYZ"


def test_cta_resolves_through_i18n_key() -> None:
    """``cta[0]`` 은 ``t("action_retry_build")`` 경유 (리터럴 직기입 금지)."""
    set_lang("ko")
    try:
        actions = chat_build_module.get_build_error_actions("boom")
        assert actions["cta"][0] == t("action_retry_build")
        assert actions["cta"][0] != "Retry Analysis"
    finally:
        set_lang("en")
    actions_en = chat_build_module.get_build_error_actions("boom")
    assert actions_en["cta"][0] == t("action_retry_build")


def test_ko_retry_build_label_byte_identical() -> None:
    """ko 라벨은 기존 노출 문자열 ``다시 시도`` 와 바이트 동일해야 한다."""
    set_lang("ko")
    try:
        label = t("action_retry_build")
    finally:
        set_lang("en")
    assert label == "다시 시도"
    assert label.encode() == "다시 시도".encode()


def test_retry_button_help_wired() -> None:
    """retry 버튼의 ``help`` 는 ``t("action_retry_build_help")``."""
    fake_st = MagicMock()
    button = _render_error_block(fake_st)
    _, kwargs = button.call_args
    assert kwargs.get("help") == t("action_retry_build_help")


def test_retry_button_enabled_when_retryable() -> None:
    """``retryable=True`` 이면 버튼이 비활성화되지 않아야 한다."""
    fake_st = MagicMock()
    button = _render_error_block(fake_st)
    _, kwargs = button.call_args
    assert kwargs.get("disabled") is False
