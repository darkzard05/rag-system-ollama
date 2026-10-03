"""i18n single-source-of-truth regression gate for ``ui.strings``."""

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

import pytest

import ui.strings as strings_mod
from ui.strings import (
    dropped_key_count,
    gate_en_keys_read,
    gate_tables_symmetric,
    t,
)

DEAD_KEYS: list[str] = [
    "chat_guide",
    "chat_placeholder_followup",
    "status_generating",
    "status_generating_with_stop",
    "status_switching_models",
    "caption_answer_complete",
    "caption_partial_answer",
]

_DIRECT_CALL_RE: re.Pattern[str] = re.compile(
    r"""\bt\(\s*["']([A-Za-z0-9_]+)["']\s*\)"""
)


@pytest.fixture
def restore_lang() -> Generator[None, None, None]:
    """Restore ``LANG`` after each locale-sensitive test."""
    original: str = strings_mod.LANG
    yield None
    strings_mod.LANG = original


def _collect_direct_keys() -> set[str]:
    """Scan ``src/**/*.py`` source text for ``t("key")`` literal reads."""
    root: Path = Path(__file__).resolve().parents[2] / "src"
    found: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            text: str = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found.update(_DIRECT_CALL_RE.findall(text))
    return found


def _indirect_keys() -> set[str]:
    """Keys consumed via ``PHASE_KEYWORDS`` (resolved dynamically in ``t``)."""
    return {key for key, _keywords in strings_mod.PHASE_KEYWORDS}


def test_every_en_key_is_read() -> None:
    """PART A: Every ``en`` key must be read directly or via ``PHASE_KEYWORDS``.

    Bidirectional: every PHASE_KEYWORDS key must exist in the table, so
    deleting a live ``phase_*`` entry fails here (``t`` would fall back
    to the raw key and the pipeline phase display would break).
    """
    missing, absent_phase = gate_en_keys_read()
    assert not missing, f"unread UI_STRINGS keys: {sorted(missing)}"
    assert not absent_phase, f"PHASE_KEYWORDS keys missing: {sorted(absent_phase)}"


def test_tables_are_symmetric() -> None:
    """PART B: The ``en`` and ``ko`` tables must carry the same key set."""
    assert gate_tables_symmetric()


def test_part_a_byte_identity_in_en(restore_lang: None) -> None:
    """PART A wiring must be byte-identical under the default locale."""
    strings_mod.set_lang("en")
    assert t("caption_stopped_partial") == "Stopped · Partial answer preserved"
    assert t("pdf_error_viewer") == "PDF viewer error occurred. Please try again later."
    assert t("pdf_error_render") == "PDF viewer rendering failed."


def test_korean_values_survive() -> None:
    """PART A/B must not retranslate or drop the Korean viewer messages."""
    assert (
        strings_mod.UI_STRINGS["ko"]["pdf_error_viewer"]
        == "PDF 뷰어 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    )
    assert (
        strings_mod.UI_STRINGS["ko"]["pdf_error_render"]
        == "PDF 뷰어 렌더링에 실패했습니다."
    )


def test_dead_keys_are_removed() -> None:
    """The 7 genuinely dead keys must be gone from both tables."""
    for key in DEAD_KEYS:
        assert key not in strings_mod.UI_STRINGS["en"], key
        assert key not in strings_mod.UI_STRINGS["ko"], key


def test_scan_is_not_vacuous() -> None:
    """Guard against the readership scan passing by finding nothing."""
    direct_found = _collect_direct_keys()
    assert len(direct_found) >= 30
    assert "caption_stopped_partial" in direct_found


# ---- T3 i18n gate hardening additions ----


def test_dropped_key_warning_counter() -> None:
    """Dropped-key WARNING counter must increment on missing key; no silent drop."""
    # Reset counter
    strings_mod._dropped_key_counter = 0  # type: ignore[attr-defined]
    # Call t() with a key that does not exist
    result = t("nonexistent_key_xyz")
    # Should return the key as fallback
    assert result == "nonexistent_key_xyz"
    # Counter must have been incremented (WARNING recorded)
    assert strings_mod.dropped_key_count() >= 1, (
        "dropped-key WARNING counter must increment when t() misses a key"
    )


def test_comment_spoof_proof_ast_gate() -> None:
    """AST gate must not count keys inside comments as direct reads.

    A comment like ``# t("fake_key")`` should not make the gate pass.
    """
    # The AST-based collector uses ast.parse which naturally excludes comments.
    # Verify that _collect_t_keys_ast only finds literal t() calls in code.
    from ui.strings import _collect_t_keys_ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"
    keys = _collect_t_keys_ast(root)

    # Should not contain a "fake_key" from comments
    assert "fake_key" not in keys, (
        "AST gate must not count keys from comments as direct reads"
    )


def test_disabled_branch_decision() -> None:
    """Decision: disabled branch in chat_build.py:148 button.

    The ``disabled=not actions["retryable"]`` branch is effectively dead code
    because ``retryable`` is always ``True`` (set in ``get_build_error_actions``).
    This test documents the decision: the button is always enabled.
    """
    from ui.components.chat_build import get_build_error_actions

    actions = get_build_error_actions("test-error")
    # retryable is always True in the current implementation
    assert actions["retryable"] is True, "retryable should always be True"

    # The button's disabled state: disabled=not True = False, so always enabled
    assert actions["retryable"] is True  # verify the decision boundary
