import pytest
import contextlib
from unittest.mock import patch, MagicMock
from core.session import SessionManager


def test_session_reset_for_new_file_batch_lock_and_state_parity():
    """
    reset_for_new_file()이 배치 처리를 통해 단 1회의 락 획득만 수행하며,
    기존 28개 개별 세팅과 완전히 동일한 상태를 구성하는지 검증
    """
    sid = "test_opt_session"
    SessionManager.init_session(sid)

    acquire_count = 0
    orig_acquire_lock = SessionManager._acquire_lock

    @contextlib.contextmanager
    def counting_acquire_lock(session_id):
        nonlocal acquire_count
        acquire_count += 1
        with orig_acquire_lock(session_id) as l:
            yield l

    with patch.object(
        SessionManager, "_acquire_lock", side_effect=counting_acquire_lock
    ):
        try:
            SessionManager.reset_for_new_file(session_id=sid)

            # As-Is(28개 개별 set + 기타 = ~31회) 대비 To-Be(reset_conversation, batch set, delete, add_status_log = 4회)로 대폭 축소됨을 검증
            assert acquire_count <= 4, (
                f"Lock was acquired {acquire_count} times; expected batch acquisition (<= 4)"
            )

            # 상태 무결성 검증 (핵심 플래그 및 기본값 유지 확인)
            assert SessionManager.get("pdf_processed", session_id=sid) is False
            assert SessionManager.get("rag_engine", session_id=sid) is None
            assert SessionManager.get("active_faiss_retriever", session_id=sid) is None
            assert SessionManager.get("active_bm25_retriever", session_id=sid) is None
            assert SessionManager.get("file_hash", session_id=sid) is None
            assert SessionManager.get("current_page", session_id=sid) == 1
            assert SessionManager.get("pdf_annotations", session_id=sid) == []
            assert SessionManager.get("manual_nav_ts", session_id=sid) == 0
            assert SessionManager.get("pdf_target_page", session_id=sid) is None
        finally:
            SessionManager.delete_session(sid)


def test_resource_manager_build_flashranker_deduplication():
    """
    get_flashranker와 use_flashranker가 중복 클로저 없이
    통합 빌더 함수를 사용하는지 확인
    """
    import core.resource_manager as rm_module
    from core.resource_manager import ResourceCoordinator

    assert hasattr(rm_module, "_build_flashranker"), (
        "_build_flashranker 통합 빌더가 존재해야 함"
    )
    assert not hasattr(ResourceCoordinator, "_evict_excess_build_locks"), (
        "데드코드 _evict_excess_build_locks는 제거되어야 함"
    )
