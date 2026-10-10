"""fix-001: 빌드 진행 폴러 마운트 게이트 검증 (3B: BuildState 단일 predicate).

``render_chat_messages_area``의 게이트 분기: BuildState가 비터미널일 때만
폴링 fragment(``_render_build_progress_fragment``)를 마운트하고,
그 외 상태(유휴/완료/에러/취소/빈 상태)는 정적 block을 렌더한다.
이 게이트가 실패하면 유휴 폴링(RERUN_FRAGMENT 폭주)이 재발한다.
"""

import os

import pytest

os.environ.setdefault("IS_CI_TEST", "true")

import ui.components.chat as chat_mod  # noqa: E402
from core.build_state import BuildPhase, initial_build_state  # noqa: E402
from core.session import SessionManager  # noqa: E402


def _seed_build_state(sid: str, phase: BuildPhase | None) -> None:
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    if phase is None:
        return
    build = initial_build_state("doc.pdf", "h1")
    build.phase = phase
    build.pct = 10 if phase not in (BuildPhase.DONE,) else 100
    SessionManager.set("build_state", build, session_id=sid)


@pytest.mark.parametrize(
    ("phase", "expect_fragment", "expect_block"),
    [
        (None, False, True),  # 빈 상태: 정적 block (early-return)
        (BuildPhase.PREPARING, True, False),
        (BuildPhase.EXTRACTING, True, False),
        (BuildPhase.CHUNKING, True, False),
        (BuildPhase.INDEXING, True, False),
        (BuildPhase.REGISTERING, True, False),
        (BuildPhase.DONE, False, True),
        (BuildPhase.ERROR, False, True),
        (BuildPhase.CANCELLED, False, True),  # 2B: 취소는 터미널 → 정적
        (BuildPhase.IDLE, False, True),
    ],
)
def test_progress_gate_mounts_poller_only_while_building(
    phase: BuildPhase | None,
    expect_fragment: bool,
    expect_block: bool,
) -> None:
    sid = "test-progress-gate"
    _seed_build_state(sid, phase)

    from unittest.mock import patch

    with (
        patch.object(chat_mod, "_render_build_progress_fragment") as frag,
        patch.object(chat_mod, "_render_build_progress_block") as block,
        patch.object(chat_mod, "_render_unified_timeline"),
    ):
        chat_mod.render_chat_messages_area()

    assert frag.called is expect_fragment
    assert block.called is expect_block
    assert frag.called != block.called, "게이트는 두 경로를 동시에 렌더하면 안 됨"


def test_progress_gate_no_double_mount_same_widget_keys() -> None:
    """정적/폴링 전환 시 키 충돌 없이 한쪽만 렌더된다(위젯 key 보존 전제)."""
    sid = "test-progress-keys"
    _seed_build_state(sid, BuildPhase.EXTRACTING)

    from unittest.mock import patch

    with (
        patch.object(chat_mod, "_render_build_progress_fragment") as frag,
        patch.object(chat_mod, "_render_build_progress_block") as block,
        patch.object(chat_mod, "_render_unified_timeline"),
    ):
        chat_mod.render_chat_messages_area()
        assert frag.call_count == 1
        assert not block.called

    # 완료 후 상태 전환 → 정적 경로 단독 렌더
    _seed_build_state(sid, BuildPhase.DONE)
    with (
        patch.object(chat_mod, "_render_build_progress_fragment") as frag2,
        patch.object(chat_mod, "_render_build_progress_block") as block2,
        patch.object(chat_mod, "_render_unified_timeline"),
    ):
        chat_mod.render_chat_messages_area()
        assert not frag2.called
        assert block2.call_count == 1


def test_build_progress_fragment_has_no_run_every() -> None:
    """run_every 재도입 금지: 데코레이터 소스에서 run_every 부재 확인.

    run_every가 있으면 프론트엔드에 지속적인 auto-rerun 타이머가 설정되어
    빌드 완료 후 전체 rerun 시 스테일 타이머가 경고를 발생시킨다.
    """
    import inspect

    source = inspect.getsource(chat_mod._render_build_progress_fragment)
    decorator_line = source.split("\n", 1)[0].strip()
    assert decorator_line.startswith("@st.fragment"), "데코레이터가 @st.fragment여야 함"
    assert "run_every" not in decorator_line, (
        "run_every 재도입 금지: 프론트엔드 지속 타이머가 스테일 fragment 경고를 유발함"
    )
