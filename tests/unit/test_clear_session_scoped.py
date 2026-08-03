"""
clear_session 격리 검증 (문제 2E).

한 세션의 초기화가 전역 리소스 풀(clear_all)을 비우지 않아야 합니다.
- 다른 세션이 같은 문서를 사용 중이면 unregister하지 않음
- 아무 세션도 문서를 참조하지 않으면 해당 문서 리트리버만 해제
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.rag_core import RAGSystem
from core.session import SessionManager


def _build_fake_resource_manager():
    rm = MagicMock()
    rm.unregister_retrievers = AsyncMock()
    rm.clear_all = AsyncMock()
    return rm


async def test_clear_session_does_not_clear_other_sessions(monkeypatch):
    SessionManager.reset()
    SessionManager.init_session("sid_a")
    SessionManager.init_session("sid_b")
    SessionManager.set("file_hash", "hash123", session_id="sid_a")
    SessionManager.set("file_hash", "hash123", session_id="sid_b")

    fake_rm = _build_fake_resource_manager()
    monkeypatch.setattr("core.rag_core.get_resource_manager", lambda: fake_rm)

    rag = RAGSystem(session_id="sid_a")
    rag.clear_session()
    await asyncio.sleep(0.05)

    fake_rm.clear_all.assert_not_called()
    # sid_b가 여전히 hash123을 사용 중이므로 unregister도 없어야 함
    fake_rm.unregister_retrievers.assert_not_called()
    assert SessionManager.get("file_hash", session_id="sid_b") == "hash123"


async def test_clear_session_unregisters_orphan_document(monkeypatch):
    SessionManager.reset()
    SessionManager.init_session("sid_c")
    SessionManager.set("file_hash", "hash_orphan", session_id="sid_c")

    fake_rm = _build_fake_resource_manager()
    monkeypatch.setattr("core.rag_core.get_resource_manager", lambda: fake_rm)

    rag = RAGSystem(session_id="sid_c")
    rag.clear_session()
    await asyncio.sleep(0.05)

    fake_rm.clear_all.assert_not_called()
    fake_rm.unregister_retrievers.assert_called_once_with("hash_orphan")
