"""
Edge-latch autoscroll tests: edge_latch(user_scrolled_up, scroll_delta_y, dist_to_bottom) -> bool.

Modules under test do not exist yet; import fails (RED).
"""

import os
import sys
from types import ModuleType

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _edge_threshold() -> int:
    from ui.components.autoscroll_edge import THRESHOLD

    return THRESHOLD


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


THRESHOLD = _edge_threshold()


def test_edge_latch_false_to_true_once() -> None:
    """E1: user_scrolled_up=False에서 한 번 true로 전환되면 latch가 true를 유지."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

        result = edge_latch(False, 0, 100)
        assert result is False
        result2 = edge_latch(True, 10, 90)
        assert result2 is True


def test_edge_latch_true_to_true_noop() -> None:
    """E2: 이미 true인 상태에서 다시 호출하면 noop(변화 없음)."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

        result = edge_latch(True, 5, THRESHOLD + 5)
        assert result is True
        result2 = edge_latch(True, 3, THRESHOLD)
        assert result2 is True


def test_edge_latch_true_to_false_reset_within_threshold() -> None:
    """E3: true에서 false로 전환되고 dist_to_bottom<=THRESHOLD이면 reset."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

    result = edge_latch(True, 0, THRESHOLD - 1)
    assert result is False


def test_edge_latch_true_to_false_reset_at_threshold() -> None:
    """E4: dist_to_bottom이 정확히 THRESHOLD이면 reset 조건 충족."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

        result = edge_latch(True, 0, THRESHOLD)
        assert result is False


def test_edge_latch_true_to_false_no_reset_beyond_threshold() -> None:
    """E5: dist_to_bottom이 THRESHOLD를 초과하면 reset되지 않음."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

        result = edge_latch(True, 0, THRESHOLD + 1)
        assert result is True


def test_edge_latch_negative_scroll_delta() -> None:
    """E6: 음수 scroll_delta_y도 정상 처리됨."""
    with _block_streamlit():
        from ui.components.autoscroll_edge import edge_latch

        result = edge_latch(False, -10, 100)
        assert result is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
