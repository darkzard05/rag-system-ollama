"""Observable contract for the document-analysis completion indicator."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from core.build_state import (  # noqa: E402
    BuildEvent,
    BuildPhase,
    initial_build_state,
)
from ui.components.doc_analysis.viewmodel import to_viewmodel  # noqa: E402
from ui.strings import set_lang, t  # noqa: E402


def _done_state(*, cache_used: bool, elapsed: int = 7):
    build = initial_build_state("contract.pdf", "hash_contract")
    build.started_at = time.time() - elapsed
    build.finished_at = build.started_at + elapsed
    build.phase = BuildPhase.DONE
    build.pct = 100
    build.cache_used = cache_used
    return build


def test_done_renders_one_terminal_view() -> None:
    set_lang("en")
    vm = to_viewmodel(_done_state(cache_used=False))
    assert vm is not None
    assert vm.status_state == "complete"
    assert vm.expanded is False
    assert vm.progress is None
    assert vm.stepper is None
    assert vm.actions == ()
    assert vm.show_error is False
    assert t("build_complete_status") in vm.label
    assert "contract.pdf" in vm.label
    assert t("build_complete_caption").format(elapsed=7) == vm.caption


def test_done_with_zero_elapsed_keeps_a_caption() -> None:
    set_lang("en")
    vm = to_viewmodel(_done_state(cache_used=False, elapsed=0))
    assert vm is not None
    assert vm.caption == t("build_complete_default")


def test_done_label_shows_elapsed_without_opening_the_status() -> None:
    set_lang("en")
    vm = to_viewmodel(_done_state(cache_used=False))
    assert vm is not None
    suffix = t("build_complete_elapsed").format(elapsed=7)
    assert suffix in vm.label


def test_zero_elapsed_label_has_no_duration_suffix() -> None:
    set_lang("en")
    vm = to_viewmodel(_done_state(cache_used=False, elapsed=0))
    assert vm is not None
    assert "0s" not in vm.label
    assert "0초" not in vm.label


def test_cache_reuse_is_visible_after_completion() -> None:
    set_lang("en")
    vm = to_viewmodel(_done_state(cache_used=True))
    assert vm is not None
    assert t("cache_tag_reused").strip() in vm.label
    assert t("cache_tag_new").strip() not in vm.label


def test_terminal_event_after_done_cannot_change_the_view() -> None:
    set_lang("en")
    build = _done_state(cache_used=False)
    before = to_viewmodel(build)
    from core.build_state import apply_event

    apply_event(
        build,
        BuildEvent(
            phase=BuildPhase.EXTRACTING, pct=10, detail_key="build_phase_extracting"
        ),
    )
    after = to_viewmodel(build)
    assert after == before


def test_error_and_cancel_never_render_complete() -> None:
    set_lang("en")
    for phase in (BuildPhase.ERROR, BuildPhase.CANCELLED):
        build = initial_build_state("contract.pdf", "hash_contract")
        build.phase = phase
        vm = to_viewmodel(build)
        assert vm is not None
        assert vm.status_state == "error"
        assert vm.actions == ("retry",)
        assert t("build_complete_status") not in vm.label


def test_idle_without_file_renders_nothing() -> None:
    set_lang("en")
    assert to_viewmodel(None) is None
