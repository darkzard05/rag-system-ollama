"""
Flag-read autoscroll tests: flag_read(is_generating_answer, active_stream_msg_id, msg_id) -> bool.

Modules under test do not exist yet; import fails (RED).
"""

import os
import sys
from types import ModuleType

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class _BlockStreamlit:
    """meta_path finder that makes ``streamlit`` (and submodules) unimportable."""

    def find_spec(self, name, path=None, target=None):  # noqa: ANN001, D401
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(f"streamlit is blocked for test: {name}")
        return None


class _no_streamlit:
    def __init__(self) -> None:
        self._saved: dict[str, ModuleType] = {}

    def __enter__(self) -> None:
        self._saved = {
            mod: sys.modules.pop(mod)
            for mod in list(sys.modules)
            if mod == "streamlit" or mod.startswith("streamlit.")
        }
        sys.meta_path.insert(0, _BlockStreamlit())

    def __exit__(self, *exc: object) -> None:
        sys.meta_path[:] = [
            f for f in sys.meta_path if not isinstance(f, _BlockStreamlit)
        ]
        for mod in list(sys.modules):
            if mod == "streamlit" or mod.startswith("streamlit."):
                sys.modules.pop(mod, None)
        sys.modules.update(self._saved)
        self._saved = {}


def _block_streamlit() -> _no_streamlit:
    return _no_streamlit()


@pytest.mark.parametrize(
    ("is_generating_answer", "active_stream_msg_id", "msg_id", "expected"),
    [
        (True, "msg_42", "msg_42", True),
        (False, "msg_42", "msg_42", False),
        (True, "msg_42", "msg_99", False),
        (False, "msg_42", "msg_99", False),
        (True, "", "", False),
        (True, "msg_1", "msg_2", False),
    ],
)
def test_flag_read(
    is_generating_answer: bool,
    active_stream_msg_id: str,
    msg_id: str,
    expected: bool,
) -> None:
    """F1-F6: flag_read returns expected value per conjunction and stale-True rule."""
    with _block_streamlit():
        from ui.components.autoscroll_flag import flag_read

        assert flag_read(is_generating_answer, active_stream_msg_id, msg_id) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
