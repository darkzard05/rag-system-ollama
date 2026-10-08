"""콘솔 핸들러 stderr 리다이렉션 추적 단위 테스트.

대상: src/common/logging_config.py::setup_logging
결함: 핸들러가 sys.__stderr__(프로세스 생성 시점 원본)에 고정되면,
  이후 sys.stderr 교체(리다이렉션/백그라운드 Hidden 기동/캡처)를
  추적하지 못해 "콘솔 로그 미출력"으로 관측된다.
기대: setup_logging 호출 시점의 sys.stderr에 바인딩되어야 한다.

실행: pytest tests/unit/test_console_handler_redirection.py -v
"""

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from common.logging_config import setup_logging


def _restore_root(handlers: list[logging.Handler], level: int) -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def test_console_handler_binds_current_stderr() -> None:
    """호출 시점의 sys.stderr에 바인딩되는지 검증한다."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        setup_logging(log_level="INFO", log_file=None, console_output=True)
        logging.getLogger("poc.console").info("hello-terminal")
        assert "hello-terminal" in captured.getvalue()
    finally:
        sys.stderr = old_stderr
        _restore_root(saved_handlers, saved_level)


def test_console_handler_ignores_dunder_stderr() -> None:
    """핸들러 스트림이 sys.__stderr__ 고정이 아님을 검증한다."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        logger = setup_logging(log_level="INFO", log_file=None, console_output=True)
        streams = [
            h.stream for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert streams, "console handler must exist"
        for stream in streams:
            assert stream is not sys.__stderr__ or sys.stderr is sys.__stderr__
    finally:
        _restore_root(saved_handlers, saved_level)
