"""
R1b-03 팬텀 상태 방지 검증 (빌드 실패 롤백 + 엔진 캐시 해시 검사).

빌드 실패 시:
(a) 세션 `file_hash`는 조기 커밋에서 롤백되어 이전 값을 유지해야 한다.
(b) `EngineCacheManager.get_engine`은 저장된 엔진의 file_hash 메타데이터와
    세션 현재 file_hash가 불일치하면 캐시를 사용하지 않아야 한다(재빌드 유도).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cache.engine_cache import EngineCacheManager
from common.exceptions import EmptyPDFError
from core.pipeline_builder import PipelineBuilder
from core.session import SessionManager


@pytest.mark.asyncio
async def test_build_failure_restores_previous_file_hash():
    """load_pdf_docs 예외 시 세션 file_hash가 이전 값으로 복원되어야 한다."""
    sid = "phantom_build_fail"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "old_hash", session_id=sid)
    SessionManager.set("rag_engine", "old_engine", session_id=sid)

    builder = PipelineBuilder(session_id=sid)

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="new_hash"),
        patch(
            "core.pipeline_builder.load_pdf_docs",
            new=AsyncMock(side_effect=RuntimeError("PDF 로드 실패")),
        ),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
    ):
        with pytest.raises(RuntimeError, match="PDF 로드 실패"):
            await builder.build(
                file_path="new.pdf",
                file_name="new.pdf",
                embedder=MagicMock(model="fake-model"),
            )

    assert SessionManager.get("file_hash", session_id=sid) == "old_hash"
    assert SessionManager.get("rag_engine", session_id=sid) == "old_engine"


@pytest.mark.asyncio
async def test_build_failure_restores_file_hash_on_vector_build_error():
    """벡터/BM25 빌드 예외 시에도 file_hash가 이전 값으로 복원되어야 한다."""
    sid = "phantom_vector_fail"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "prev_hash", session_id=sid)

    builder = PipelineBuilder(session_id=sid)

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="next_hash"),
        patch(
            "core.pipeline_builder.load_pdf_docs",
            new=AsyncMock(return_value=["doc"]),
        ),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=(["chunk"], None)),
        ),
        patch(
            "core.pipeline_builder.create_vector_store",
            side_effect=RuntimeError("벡터 빌드 실패"),
        ),
        patch("core.pipeline_builder.create_bm25_retriever", return_value="bm25"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
    ):
        with pytest.raises(RuntimeError, match="벡터 빌드 실패"):
            await builder.build(
                file_path="new.pdf",
                file_name="new.pdf",
                embedder=MagicMock(model="fake-model"),
            )

    assert SessionManager.get("file_hash", session_id=sid) == "prev_hash"


@pytest.mark.asyncio
async def test_build_failure_on_fresh_session_leaves_no_file_hash():
    """이전 문서가 없는 세션에서 빌드 실패 시 file_hash가 남지 않아야 한다."""
    sid = "phantom_fresh_fail"
    SessionManager.reset()
    SessionManager.init_session(sid)

    builder = PipelineBuilder(session_id=sid)

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="new_hash"),
        patch(
            "core.pipeline_builder.load_pdf_docs",
            new=AsyncMock(side_effect=EmptyPDFError()),
        ),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
    ):
        with pytest.raises(EmptyPDFError):
            await builder.build(
                file_path="new.pdf",
                file_name="new.pdf",
                embedder=MagicMock(model="fake-model"),
            )

    assert SessionManager.get("file_hash", session_id=sid) is None


@pytest.mark.asyncio
async def test_build_success_keeps_hash_and_syncs_engine_metadata():
    """성공 시 새 file_hash가 유지되고 엔진 해시 메타데이터가 일치해야 한다."""
    sid = "phantom_build_success"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "old_hash", session_id=sid)

    builder = PipelineBuilder(session_id=sid)

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="new_hash"),
        patch(
            "core.pipeline_builder.load_pdf_docs",
            new=AsyncMock(return_value=["doc"]),
        ),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=(["chunk"], None)),
        ),
        patch("core.pipeline_builder.create_vector_store", return_value="vs"),
        patch("core.pipeline_builder.create_bm25_retriever", return_value="bm25"),
        patch("core.pipeline_builder.get_resource_manager") as mock_rm,
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value="workflow"),
        ),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
    ):
        rm = MagicMock()
        rm.register_retrievers = AsyncMock(return_value=None)
        mock_rm.return_value = rm
        embedder = MagicMock()
        embedder.model = "fake-model"
        embedder.model_name = "fake-model"

        msg, cache_used = await builder.build(
            file_path="new.pdf", file_name="new.pdf", embedder=embedder
        )

    assert "신규 인덱싱 완료" in msg
    assert not cache_used
    assert SessionManager.get("file_hash", session_id=sid) == "new_hash"
    assert SessionManager.get("rag_engine", session_id=sid) == "workflow"
    assert SessionManager.get("rag_engine_file_hash", session_id=sid) == "new_hash"


def test_get_engine_invalidates_on_file_hash_mismatch():
    """저장된 엔진의 해시와 세션 해시 불일치 시 캐시를 사용하지 않아야 한다."""
    sid = "phantom_engine_hash_mismatch"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()
    SessionManager.set("file_hash", "hash_a", session_id=sid)

    async def _run():
        EngineCacheManager.set_engine(sid, engine)
        # 동일 해시 → 기존 엔진 재사용
        assert EngineCacheManager.get_engine(sid) is engine
        # 새 문서 해시로 변경 → 이전 엔진 반환 금지 (재빌드 유도)
        SessionManager.set("file_hash", "hash_b", session_id=sid)
        assert EngineCacheManager.get_engine(sid) is None

    asyncio.run(_run())
