"""A skipped E2E check must not fail the integrity run."""

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "maintenance"
    / "verify_integrity.py"
)
_spec = importlib.util.spec_from_file_location("verify_integrity", _SCRIPT)
assert _spec is not None and _spec.loader is not None
verify_integrity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_integrity)


def test_offline_ollama_does_not_fail_integrity() -> None:
    output = io.StringIO()
    with (
        patch.object(verify_integrity, "check_ollama", return_value=False),
        patch.object(verify_integrity, "run_command", return_value=True),
        patch.object(
            verify_integrity, "run_query_latency_regression", return_value=True
        ),
        patch.object(sys, "exit") as exit_mock,
        redirect_stdout(output),
    ):
        verify_integrity.main()

    exit_mock.assert_not_called()
    assert "SKIP" in output.getvalue()
