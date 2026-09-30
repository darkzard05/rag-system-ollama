"""
Stick-math autoscroll tests: stick_math(scroll_height, scroll_top, client_height, threshold=BOTTOM_STICK_PX) -> dict[str, bool | int].

Modules under test do not exist yet; import fails (RED).
"""

import os
import sys
from types import ModuleType

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _default_threshold() -> int:
    from ui.components.autoscroll_stick import BOTTOM_STICK_PX

    return BOTTOM_STICK_PX


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


THRESHOLD = _default_threshold()


def test_stick_math_at_zero_scroll() -> None:
    """S1: scroll_top=0일 때 dist 계산 및 기본 반환값."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(1000, 0, 800, THRESHOLD)
        assert isinstance(result, dict)
        assert "dist" in result
        assert result["dist"] == 1000 - 0 - 800


def test_stick_math_just_below_threshold() -> None:
    """S2: dist가 임계값 1px 미만이면 stick_active=True."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        dist = THRESHOLD - 1
        result = stick_math(1000, 1000 - dist - 800, 800, THRESHOLD)
        assert result["dist"] == dist
        assert result["stick_active"] is True


def test_stick_math_at_exact_threshold() -> None:
    """S3: dist가 정확히 임계값이면 stick_active=True."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(1000, 1000 - THRESHOLD - 800, 800, THRESHOLD)
        assert result["dist"] == THRESHOLD
        assert result["stick_active"] is True


def test_stick_math_just_above_threshold() -> None:
    """S4: dist가 임계값 1px 초과면 stick_active=False."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        dist = THRESHOLD + 1
        result = stick_math(1000, 1000 - dist - 800, 800, THRESHOLD)
        assert result["dist"] == dist
        assert result["stick_active"] is False


def test_stick_math_dist_formula() -> None:
    """S5: dist = scrollHeight - scrollTop - clientHeight 공식 검증."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(2000, 500, 600, THRESHOLD)
        assert result["dist"] == 2000 - 500 - 600


def test_stick_math_fallback_dist_negative_one() -> None:
    """S6: 유효하지 않은 입력 시 dist=-1로 폴백."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(0, 0, 0, THRESHOLD)
        assert result["dist"] == -1


def test_stick_math_default_threshold() -> None:
    """S7: threshold 미지정 시 BOTTOM_STICK_PX 기본값 사용."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(1000, 0, 800)
        assert result["dist"] == 200


def test_stick_math_large_scroll() -> None:
    """S8: 큰 scroll_height와 scroll_top에서도 정상 동작."""
    with _block_streamlit():
        from ui.components.autoscroll_stick import stick_math

        result = stick_math(10000, 9000, 800, THRESHOLD)
        assert result["dist"] == 200
        assert isinstance(result["stick_active"], bool)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
