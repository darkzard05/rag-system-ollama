"""A late failure must not turn an already completed build back into ERROR."""

from __future__ import annotations

from core.build_state import (
    BuildPhase,
    complete_build,
    fail_build,
    initial_build_state,
)


def test_late_failure_keeps_done_phase() -> None:
    state = complete_build(initial_build_state("doc.pdf", "hash1"), cache_used=False)

    fail_build(state, "late cleanup failure")

    assert state.phase is BuildPhase.DONE
    assert state.error is None


def test_late_completion_keeps_error_phase() -> None:
    state = fail_build(initial_build_state("doc.pdf", "hash1"), "build failed")

    complete_build(state, cache_used=False)

    assert state.phase is BuildPhase.ERROR
    assert state.error == "build failed"
