from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.config import DEFAULT_EMBEDDING_MODEL
from core.pipeline_builder import prepare_query_config_or_build
from core.session import SessionManager


@pytest.mark.asyncio
async def test_prepare_query_config_model_and_embedder_caching():
    sid = "test-session-model-switch"
    SessionManager.reset_all_state(sid)

    mock_llm_a = MagicMock()
    mock_llm_a.model_name = "model_a"
    mock_llm_b = MagicMock()
    mock_llm_b.model_name = "model_b"
    mock_llm_c = MagicMock()
    mock_llm_c.model_name = "model_c"

    mock_emb_default = MagicMock()
    mock_emb_default.model_name = DEFAULT_EMBEDDING_MODEL
    mock_emb_new = MagicMock()
    mock_emb_new.model_name = "new_emb"

    with (
        patch(
            "core.model_loader.ModelManager.get_llm", new_callable=AsyncMock
        ) as mock_get_llm,
        patch(
            "core.model_loader.ModelManager.get_embedder", new_callable=AsyncMock
        ) as mock_get_embedder,
        patch("core.pipeline_builder.get_resource_manager") as mock_get_rm,
    ):
        mock_get_llm.side_effect = lambda m: {
            "model_a": mock_llm_a,
            "model_b": mock_llm_b,
            "model_c": mock_llm_c,
        }.get(m, mock_llm_a)
        mock_get_embedder.side_effect = (
            lambda e: mock_emb_new if e == "new_emb" else mock_emb_default
        )

        rm = MagicMock()
        rm.retrievers.get.return_value = (MagicMock(), MagicMock())
        rm.get_retrievers = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_get_rm.return_value = rm

        SessionManager.set("file_hash", "hash_test", session_id=sid)

        # 1st call with model_a
        cfg1 = await prepare_query_config_or_build(sid, model_name="model_a")
        assert cfg1["configurable"]["llm"] == mock_llm_a
        assert SessionManager.get("llm_model_name", session_id=sid) == "model_a"
        assert mock_get_llm.call_count == 1
        assert mock_get_embedder.call_count == 1

        # 2nd call with same model_a (should be cached)
        cfg2 = await prepare_query_config_or_build(sid, model_name="model_a")
        assert cfg2["configurable"]["llm"] == mock_llm_a
        assert mock_get_llm.call_count == 1  # No extra call

        # 3rd call with model_b (should reload llm)
        cfg3 = await prepare_query_config_or_build(sid, model_name="model_b")
        assert cfg3["configurable"]["llm"] == mock_llm_b
        assert SessionManager.get("llm_model_name", session_id=sid) == "model_b"
        assert mock_get_llm.call_count == 2

        # 4th call with model_name=None but last_selected_model = model_c
        SessionManager.set("last_selected_model", "model_c", session_id=sid)
        cfg4 = await prepare_query_config_or_build(sid, model_name=None)
        assert cfg4["configurable"]["llm"] == mock_llm_c
        assert SessionManager.get("llm_model_name", session_id=sid) == "model_c"
        assert mock_get_llm.call_count == 3

        # 5th call: embedding model change
        SessionManager.set("last_selected_embedding_model", "new_emb", session_id=sid)
        await prepare_query_config_or_build(sid, model_name="model_c")
        assert SessionManager.get("embedder_model_name", session_id=sid) == "new_emb"
        assert mock_get_embedder.call_count == 2

    SessionManager.reset_all_state(sid)


@pytest.mark.asyncio
async def test_manual_llm_injection_backward_compatibility():
    sid = "test-session-manual"
    SessionManager.reset_all_state(sid)

    fake_llm = MagicMock()
    fake_llm.model_name = "custom_injected"
    SessionManager.set("llm", fake_llm, session_id=sid)
    SessionManager.set("embedder", MagicMock(), session_id=sid)
    SessionManager.set("file_hash", "hash_test", session_id=sid)

    with (
        patch(
            "core.model_loader.ModelManager.get_llm", new_callable=AsyncMock
        ) as mock_get_llm,
        patch("core.pipeline_builder.get_resource_manager") as mock_get_rm,
    ):
        rm = MagicMock()
        rm.retrievers.get.return_value = (MagicMock(), MagicMock())
        rm.get_retrievers = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_get_rm.return_value = rm

        cfg = await prepare_query_config_or_build(sid, model_name=None)
        assert cfg["configurable"]["llm"] == fake_llm
        mock_get_llm.assert_not_called()

    SessionManager.reset_all_state(sid)
