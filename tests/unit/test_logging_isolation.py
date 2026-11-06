"""로그 경로 격리 회귀 테스트.

pytest 실행 중에는 운영 ``logs/app.log`` 대신 임시 파일을 사용해야 한다.
테스트가 운영 로그에 예외를 기록하면 로그 기반 장애 분석이 불가능해진다.
"""

from pathlib import Path

import pytest

from main import FilePathConstants, _resolve_log_file


def test_pytest_uses_isolated_log_file() -> None:
    """conftest 가 설정한 pytest 로그 경로가 운영 로그와 달라야 한다."""
    resolved = _resolve_log_file()

    assert resolved is not None
    assert resolved != FilePathConstants.LOG_FILE
    assert resolved.parent.name.startswith("rag_ollama_logs_")


def test_explicit_log_file_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RAG_LOG_FILE 로 경로를 재지정할 수 있어야 한다."""
    target = tmp_path / "custom.log"
    monkeypatch.setenv("RAG_LOG_FILE", str(target))
    monkeypatch.delenv("RAG_LOG_DISABLE_FILE", raising=False)

    assert _resolve_log_file() == target


def test_disable_file_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG_LOG_DISABLE_FILE=1 이면 파일 로깅을 끈다."""
    monkeypatch.setenv("RAG_LOG_DISABLE_FILE", "1")

    assert _resolve_log_file() is None
