import main
import ui.components.chat as chat_mod
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.session import SessionManager


def test_red_unused_main_functions_removed():
    """main.py에서 미사용 동기 _update_qa_chain이 제거되었는지 검증"""
    assert not hasattr(main, "_update_qa_chain"), (
        "_update_qa_chain should be removed as dead code"
    )


def test_red_deprecated_rebuild_keys_removed_from_session():
    """SessionManager에서 사장된 rebuild 플랫 키들이 제거되었는지 검증"""
    sid = "test_cleanup_session"
    SessionManager.init_session(sid)
    state = SessionManager._fallback_sessions[sid]

    deprecated_keys = {"rebuild_progress", "rebuild_status", "rebuild_done"}
    for key in deprecated_keys:
        assert key not in SessionManager.DEFAULT_SESSION_STATE
        assert key not in state
        assert key not in state["_dirty_keys"]

    SessionManager.reset_for_new_file(sid)
    for key in deprecated_keys:
        assert SessionManager.get(key, session_id=sid) is None


def test_red_semantic_chunker_is_flattened():
    """EmbeddingBasedSemanticChunker가 믹스인 없이 단일 클래스로 구성되었는지 검증"""
    bases = EmbeddingBasedSemanticChunker.__bases__
    base_names = [b.__name__ for b in bases]
    assert "SemanticChunkerSentencesMixin" not in base_names
    assert "SemanticChunkerEmbeddingsMixin" not in base_names
    assert "SemanticChunkerBreakpointsMixin" not in base_names
    assert "SemanticChunkerMergeMixin" not in base_names
    assert "SemanticChunkerMetadataMixin" not in base_names
    assert bases == (object,)


def test_red_chat_dead_globals_removed():
    """chat.py에서 불필요한 _PHASE_LABELS 모듈 전역 변수가 제거되었는지 검증"""
    assert not hasattr(chat_mod, "_PHASE_LABELS"), (
        "_PHASE_LABELS global variable should be removed"
    )
