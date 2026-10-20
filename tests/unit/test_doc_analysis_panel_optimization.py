"""Regression tests for the document analysis panel render-path optimizations."""

from unittest.mock import patch

import pytest

from core.build_state import BuildPhase, complete_build, initial_build_state
from core.session import SessionManager
from ui.components.doc_analysis.panel import _doc_cache_override
from ui.components.doc_analysis.viewmodel import to_viewmodel
from ui.strings import set_lang

SID = "test_doc_panel_opt_sid"


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.init_session(SID)
    yield
    SessionManager.reset()
    set_lang("en")


def test_doc_cache_override_skips_session_get_when_cache_used_present():
    """A resolved build.cache_used must bypass the session lock entirely."""
    build = initial_build_state("test.pdf", "hash1")
    complete_build(build, cache_used=True)
    assert build.cache_used is True

    with patch.object(SessionManager, "get") as mock_get:
        result = _doc_cache_override(SID, build)

    assert result is True
    mock_get.assert_not_called()


def test_doc_cache_override_falls_back_to_doc_stats_when_unresolved():
    """When build.cache_used is unset, the fresh doc_stats flag still wins."""
    build = initial_build_state("test.pdf", "hash1")
    assert build.cache_used is None
    SessionManager.set("doc_stats", {"cache_used": False}, SID)

    assert _doc_cache_override(SID, build) is False


def test_to_viewmodel_done_label_i18n_support():
    """The DONE label follows the active locale (ko: 준비 완료, en: Ready)."""
    build = initial_build_state("doc.pdf", "hash1")
    complete_build(build, cache_used=False)
    assert build.phase == BuildPhase.DONE

    set_lang("ko")
    vm_ko = to_viewmodel(build, cache_used=False)
    assert vm_ko is not None
    assert "준비 완료" in vm_ko.label

    set_lang("en")
    vm_en = to_viewmodel(build, cache_used=False)
    assert vm_en is not None
    assert "Ready" in vm_en.label
    assert "준비 완료" not in vm_en.label
