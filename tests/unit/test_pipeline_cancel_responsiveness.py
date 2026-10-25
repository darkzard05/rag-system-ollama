"""PipelineBuilder 취소 반응성 회귀 테스트.

텍스트 추출 직후(고비용 청킹/임베딩 진입 전) 취소 시그널이 도달하면
``split_documents``가 호출되지 않고 즉시 ``asyncio.CancelledError``로
종료되어야 합니다.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_pipeline_aborts_immediately_before_split_documents():
    sid = "test_cancel_early_sid"
    cancel_event = asyncio.Event()

    async def fake_load_pdf(*args, **kwargs):
        # 텍스트 추출 완료 후 사용자가 취소 버튼을 누른 상황 시뮬레이션
        cancel_event.set()
        return [Document(page_content="Content " * 100, metadata={"page": 1})]

    mock_split = AsyncMock(return_value=([Document(page_content="chunk")], []))
    embedder = MagicMock()

    builder = PipelineBuilder(session_id=sid)

    with (
        patch(
            "core.pipeline_builder.compute_file_hash", return_value="hash_cancel_test"
        ),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch("core.pipeline_builder.load_pdf_docs", side_effect=fake_load_pdf),
        patch("core.pipeline_builder.split_documents", new=mock_split),
        pytest.raises(asyncio.CancelledError),
    ):
        await builder.build(
            file_path="dummy.pdf",
            file_name="dummy.pdf",
            embedder=embedder,
            cancel=cancel_event,
        )

    # [GREEN 조건] 추출 직후 취소되었으므로 고비용 청킹/임베딩이 호출되지 않아야 함
    mock_split.assert_not_called()
