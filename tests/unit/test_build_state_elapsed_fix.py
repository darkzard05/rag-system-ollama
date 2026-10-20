"""Fix: elapsed_sec은 터미널 전이 시 finished_at 기준으로 고정되어야 한다."""

from __future__ import annotations

import time

from core.build_state import (
    BuildPhase,
    complete_build,
    elapsed_sec,
    fail_build,
    initial_build_state,
)


def test_elapsed_sec_active_build_advances() -> None:
    """진행 중인 빌드는 현재 시간에 따라 경과 초가 계산되는지 검증."""
    state = initial_build_state("test.pdf", "hash1")
    t0 = time.time() - 5.0
    state.started_at = t0

    elapsed = elapsed_sec(state, now=t0 + 5.2)
    assert elapsed == 5


def test_elapsed_sec_terminal_build_remains_frozen() -> None:
    """완료 후 시간이 지나도 finished_at - started_at으로 고정되는지 검증."""
    state = initial_build_state("test.pdf", "hash1")
    t0 = 1000.0
    state.started_at = t0
    state.finished_at = t0 + 4.8
    state.phase = BuildPhase.DONE

    elapsed_later = elapsed_sec(state, now=t0 + 3600.0)
    assert elapsed_later == 4, f"소요 시간이 고정되지 않고 증가함: {elapsed_later}s"


def test_elapsed_sec_fail_build_remains_frozen() -> None:
    """실패(ERROR) 상태에서도 finished_at 기준으로 소요 시간이 고정되는지 검증."""
    state = initial_build_state("test.pdf", "hash1")
    state.started_at = 2000.0
    fail_build(state, "파싱 오류")
    state.finished_at = 2000.0 + 3.0

    elapsed = elapsed_sec(state, now=2000.0 + 100.0)
    assert elapsed == 3


def test_elapsed_sec_complete_build_without_explicit_finished() -> None:
    """complete_build가 기록한 finished_at만으로도 고정 소요 시간이 산출된다."""
    state = initial_build_state("test.pdf", "hash1")
    state.started_at = 5000.0
    state.finished_at = None
    complete_build(state, cache_used=False)

    assert state.finished_at is not None
    frozen = elapsed_sec(state)
    assert elapsed_sec(state, now=state.finished_at + 10_000.0) == frozen
