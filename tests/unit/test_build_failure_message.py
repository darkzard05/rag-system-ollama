"""Build failure messages must be actionable for known causes."""

import pytest

from common.exceptions import EmptyPDFError
from main import _build_failure_message


def test_empty_pdf_message_is_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui import strings

    monkeypatch.setattr(strings, "LANG", "en")
    msg = _build_failure_message(EmptyPDFError("scanned.pdf"))
    assert "no extractable text" in msg
    assert "OCR" in msg


def test_empty_pdf_message_localized(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui import strings

    monkeypatch.setattr(strings, "LANG", "ko")
    msg = _build_failure_message(EmptyPDFError("scanned.pdf"))
    assert "스캔" in msg


def test_unknown_failure_falls_back_to_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ui import strings

    monkeypatch.setattr(strings, "LANG", "en")
    msg = _build_failure_message(ValueError("boom"))
    assert (
        msg
        == "문서 분석 중 오류가 발생했습니다. 파일 상태를 확인하신 후 다시 업로드해 주세요."
    )
