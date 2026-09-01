"""fix-001: 빌드 진행 폴러 마운트 게이트 검증.

``render_chat_messages_area``의 게이트 분기: 빌드/취소 진행 중에만
1.5s 폴링 fragment(``_render_build_progress_fragment``)를 마운트하고,
그 외 상태(유휴/완료/에러/빈 상태)는 정적 block(``_render_build_progress_block``)을
렌더한다. 이 게이트가 실패하면 유휴 폴링(RERUN_FRAGMENT 폭주)이 재발한다.
"""

import os

import pytest

os.environ.setdefault("IS_CI_TEST", "true")

import ui.components.chat as chat_mod  # noqa: E402
from core.session import SessionManager  # noqa: E402

_EMPTY: dict[str, object] = {}


@pytest.mark.parametrize(
    ("session_values", "expect_fragment", "expect_block"),
    [
        (_EMPTY, False, True),  # 빈 상태: 정적 block (early-return)
        ({"is_building_rag": True}, True, False),  # 빌드 진행 중: 폴러
        ({"rebuild_cancelled": True}, True, False),  # 취소 진행 중: 폴러 유지
        ({"is_building_rag": False, "rebuild_cancelled": False}, False, True),
        ({"is_building_rag": True, "rebuild_cancelled": True}, True, False),
    ],
)
def test_progress_gate_mounts_poller_only_while_building(
    session_values: dict[str, object],
    expect_fragment: bool,
    expect_block: bool,
) -> None:
    sid = "test-progress-gate"
    SessionManager.set_session_id(sid)
    for key, value in session_values.items():
        SessionManager.set(key, value, session_id=sid)

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
    SessionManager.set_session_id(sid)
    SessionManager.set("is_building_rag", True, session_id=sid)

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
    SessionManager.set("is_building_rag", False, session_id=sid)
    SessionManager.set("rebuild_cancelled", False, session_id=sid)
    with (
        patch.object(chat_mod, "_render_build_progress_fragment") as frag2,
        patch.object(chat_mod, "_render_build_progress_block") as block2,
        patch.object(chat_mod, "_render_unified_timeline"),
    ):
        chat_mod.render_chat_messages_area()
        assert not frag2.called
        assert block2.call_count == 1
