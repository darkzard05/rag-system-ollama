"""Phase C 회귀 테스트 — 세션 스냅샷, keep_alive 단축, 중단 경로 자원 회수.

커버리지:
- C1: ``SessionManager.get_build_state``가 live 참조가 아닌 스냅샷을 반환.
- C2: ``OLLAMA_KEEP_ALIVE`` 기본값 5분 + ``_keep_alive_seconds`` 폴백 300초.
- C3: ``_drain_hydration_tasks(cancel_pending=True)``가 pending을 취소하고
  짧은 상한 안에 회수, 중단 시 하이드레이션 예외를 스트림 실패로 승격하지 않음.
- C4: UIBridge가 세션별로 강제 동기화 상태를 분리 추적(다른 파일에 병합).
"""

import asyncio

import pytest


def test_get_build_state_returns_snapshot() -> None:
    """C1: 반환된 BuildState 변이가 저장소 SSoT를 오염시키지 않아야 한다."""
    from core.build_state import BuildPhase
    from core.session import SessionManager

    sid = "c1-snapshot-sid"
    SessionManager.init_session(sid)
    created = SessionManager.init_build_state("doc.pdf", "hash-c1", session_id=sid)

    first = SessionManager.get_build_state(session_id=sid)
    assert first is not None
    assert first is not created
    assert first.phase == created.phase

    # 외부 보유 참조의 락-밖 직접 변이는 SSoT에 반영되면 안 된다.
    first.phase = BuildPhase.ERROR
    first.error = "tampered"

    second = SessionManager.get_build_state(session_id=sid)
    assert second is not None
    assert second is not first
    assert second.phase == created.phase
    assert second.error is None

    SessionManager.delete_session(sid)


def test_get_build_state_none_when_absent() -> None:
    """C1: 빌드 상태가 없으면 None."""
    from core.session import SessionManager

    sid = "c1-absent-sid"
    SessionManager.init_session(sid)
    assert SessionManager.get_build_state(session_id=sid) is None
    SessionManager.delete_session(sid)


def test_keep_alive_seconds_default_is_five_minutes(monkeypatch) -> None:
    """C2: '5m' → 300초."""
    import core._loaders as loaders

    monkeypatch.setattr(loaders, "OLLAMA_KEEP_ALIVE", "5m")
    assert loaders._keep_alive_seconds() == 300


def test_keep_alive_seconds_fallback_is_five_minutes(monkeypatch) -> None:
    """C2: 파싱 불가 값은 30분이 아니라 5분으로 폴백."""
    import core._loaders as loaders

    monkeypatch.setattr(loaders, "OLLAMA_KEEP_ALIVE", "not-a-duration")
    assert loaders._keep_alive_seconds() == 300


def test_keep_alive_seconds_honours_env_minutes(monkeypatch) -> None:
    """C2: 명시적 '30m'은 여전히 1800초(재정의 가능)."""
    import core._loaders as loaders

    monkeypatch.setattr(loaders, "OLLAMA_KEEP_ALIVE", "30m")
    assert loaders._keep_alive_seconds() == 1800


def test_config_yml_keep_alive_is_five_minutes() -> None:
    """C2: 배포 config.yml의 keep_alive가 5m로 단축됐는지 확인."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / "config.yml").read_text(encoding="utf-8"))
    assert data["models"]["keep_alive"] == "5m"


@pytest.mark.asyncio
async def test_drain_cancels_pending_when_aborted() -> None:
    """C3: cancel_pending=True면 pending 태스크를 취소하고 빠르게 회수한다."""
    from core.rag_core import _drain_hydration_tasks

    started = asyncio.Event()

    async def _hang() -> None:
        started.set()
        await asyncio.sleep(30)

    tasks = [asyncio.create_task(_hang())]
    await started.wait()

    t0 = asyncio.get_event_loop().time()
    drain = await _drain_hydration_tasks(
        tasks,
        timeout=1.0,
        msg_id="c3-abort",
        cancel_pending=True,
    )
    elapsed = asyncio.get_event_loop().time() - t0

    assert elapsed < 2.0, "cancelled tasks must be reaped quickly"
    assert drain.timed_out is False
    assert tasks[0].cancelled()


@pytest.mark.asyncio
async def test_drain_without_cancel_waits_for_completion() -> None:
    """C3: 정상 경로(cancel_pending=False)는 완료를 기다리고 결과를 보존한다."""
    from core.rag_core import _drain_hydration_tasks

    async def _work() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    tasks = [asyncio.create_task(_work())]
    drain = await _drain_hydration_tasks(tasks, timeout=5.0, msg_id="c3-normal")
    assert drain.failed == 0
    assert drain.timed_out is False
    assert drain.results == ["ok"]
