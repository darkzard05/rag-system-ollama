"""문서 분석(빌드) 진행 상태의 단일 진실원천(SSoT) 타입.

평탄 세션 키(``rebuild_progress``/``rebuild_status``/``rebuild_done``/
``rebuild_cancelled`` 등 10여 개 분산) 대체를 위한 타입 계약 모듈이다.
순수 타입 + 순수 함수만 포함하며 Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class BuildPhase(str, Enum):
    """빌드 파이프라인 위상. 터미널은 DONE/ERROR/CANCELLED."""

    IDLE = "idle"
    PREPARING = "preparing"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    REGISTERING = "registering"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_PHASES: frozenset[BuildPhase] = frozenset(
    {BuildPhase.DONE, BuildPhase.ERROR, BuildPhase.CANCELLED}
)

# 단계별 대표 진행률 (파이프라인 emit 단일 테이블 — pct 매직넘버 분산 금지).
PHASE_PCT: dict[BuildPhase, int] = {
    BuildPhase.IDLE: 0,
    BuildPhase.PREPARING: 0,
    BuildPhase.EXTRACTING: 25,
    BuildPhase.CHUNKING: 55,
    BuildPhase.INDEXING: 75,
    BuildPhase.REGISTERING: 90,
    BuildPhase.DONE: 100,
    BuildPhase.ERROR: 0,
    BuildPhase.CANCELLED: 0,
}

# 위상별 i18n detail 키 (strings.py가 소유, 여기는 참조만).
PHASE_DETAIL_KEY: dict[BuildPhase, str] = {
    BuildPhase.IDLE: "build_in_progress",
    BuildPhase.PREPARING: "build_in_progress",
    BuildPhase.EXTRACTING: "build_phase_extracting",
    BuildPhase.CHUNKING: "build_phase_chunking",
    BuildPhase.INDEXING: "build_phase_indexing",
    BuildPhase.REGISTERING: "build_phase_registering",
    BuildPhase.DONE: "build_complete_default",
    BuildPhase.ERROR: "build_phase_error",
    BuildPhase.CANCELLED: "build_cancelled",
}


@dataclass(frozen=True)
class BuildEvent:
    """파이프라인이 발행하는 타입 단계 이벤트 (pct 정수 대체).

    ``detail_key``는 P2 UI가 ``t()``로 해석하는 i18n 키, ``status``는
    P2 교체 전까지 구 UI(문자열 직접 표시)용 표시문이다.
    """

    phase: BuildPhase
    pct: int
    detail_key: str = ""
    status: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class BuildState:
    """세션당 단일 빌드 상태 객체."""

    phase: BuildPhase = BuildPhase.IDLE
    pct: int = 0
    file_name: str = ""
    file_hash: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    cache_used: bool | None = None
    error: str | None = None
    attempt: int = 0


@dataclass(frozen=True)
class BuildResult:
    """빌드 완료 결과 (페이지/청크 수는 의도적으로 제외 — 결정 1)."""

    cache_used: bool
    duration_s: float


def initial_build_state(
    file_name: str, file_hash: str = "", attempt: int = 0
) -> BuildState:
    """새 분석 시작용 초기 상태 (PREPARING, 0%)."""
    return BuildState(
        phase=BuildPhase.PREPARING,
        pct=0,
        file_name=file_name,
        file_hash=file_hash,
        started_at=time.time(),
        finished_at=None,
        cache_used=None,
        error=None,
        attempt=attempt,
    )


def is_terminal(phase: BuildPhase) -> bool:
    """터미널 위상(DONE/ERROR/CANCELLED) 여부."""
    return phase in TERMINAL_PHASES


def apply_event(state: BuildState, event: BuildEvent) -> BuildState:
    """이벤트를 상태에 반영한다 (제자리 변이 후 동일 객체 반환).

    단조성: 실행 중 위상의 pct가 현재보다 낮으면 무시한다 (역행 방지).
    터미널 이벤트는 finished_at을 확정한다.
    """
    if is_terminal(state.phase):
        return state
    if not is_terminal(event.phase) and event.pct < state.pct:
        return state
    state.phase = event.phase
    state.pct = max(0, min(100, event.pct))
    if is_terminal(event.phase) and state.finished_at is None:
        state.finished_at = event.ts or time.time()
    return state


def fail_build(state: BuildState, error: str) -> BuildState:
    """에러 터미널 전이."""
    state.phase = BuildPhase.ERROR
    state.error = error
    if state.finished_at is None:
        state.finished_at = time.time()
    return state


def cancel_build(state: BuildState) -> BuildState:
    """취소 터미널 전이."""
    if is_terminal(state.phase):
        return state
    state.phase = BuildPhase.CANCELLED
    if state.finished_at is None:
        state.finished_at = time.time()
    return state


def complete_build(
    state: BuildState, *, cache_used: bool, pct: int = 100
) -> BuildState:
    """완료 터미널 전이."""
    state.phase = BuildPhase.DONE
    state.pct = max(0, min(100, pct))
    state.cache_used = cache_used
    state.error = None
    if state.finished_at is None:
        state.finished_at = time.time()
    return state


def elapsed_sec(state: BuildState, now: float | None = None) -> int:
    """시작 기준 경과 초 (시작 시각 없으면 0)."""
    if not state.started_at:
        return 0
    ref = now if now is not None else time.time()
    return max(0, int(ref - state.started_at))
