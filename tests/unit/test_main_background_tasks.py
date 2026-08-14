# src/main.py의 백그라운드 스레드 작업 예외 처리를 검증하는 테스트
import asyncio
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath("src"))

from core.document_processor import compute_file_hash
from core.session import SessionManager
from main import _bg_rebuild_task, _update_qa_chain, on_file_upload
from ui.components.streaming import stream_chunks


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _make_test_pdf(text: str = "test") -> bytes:
    """fitz로 최소한의 유효한 PDF 바이트를 생성합니다 (upload 검증 게이트 통과용)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class TestMainBackgroundTasks(unittest.TestCase):
    def setUp(self):
        SessionManager.reset_all_state("test_session")
        SessionManager.set_session_id("test_session")

    @patch("infra.notification_system.SystemNotifier.error")
    @patch("core.rag_core.RAGSystem.build_pipeline")
    def test_rebuild_rag_system_exception_handling(
        self, mock_build, mock_notifier_error
    ):
        """RAG 빌드 중 예외 발생 시 세션 상태에 에러가 기록되는지 검증"""

        # 1. 환경 설정
        mock_build.side_effect = Exception("RAG Build Failed Mock Error")

        SessionManager.set("last_uploaded_file_name", "test.pdf")
        SessionManager.set("pdf_file_path", "test.pdf")

        # 2. 실행
        asyncio.run(
            _bg_rebuild_task(
                session_id="test_session",
                file_path="test.pdf",
                file_name="test.pdf",
                embedder_name="test-embedder",
            )
        )

        # 3. 검증
        assert not SessionManager.get("pdf_processed", session_id="test_session")
        error_msg = SessionManager.get(
            "pdf_processing_error", session_id="test_session"
        )
        assert "RAG Build Failed Mock Error" in error_msg

        # 시스템 메시지에 에러가 추가되었는지 확인
        messages = SessionManager.get_messages(session_id="test_session")
        assert any("RAG Build Failed Mock Error" in m["content"] for m in messages)

    def test_on_file_upload_same_name_different_hash_triggers_new_analysis(self):
        old_bytes = _make_test_pdf("old")
        new_bytes = _make_test_pdf("new")

        SessionManager.set(
            "last_uploaded_file_name", "test.pdf", session_id="test_session"
        )
        SessionManager.set(
            "file_hash",
            compute_file_hash("", data=old_bytes),
            session_id="test_session",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            old_file_path = Path(temp_dir) / "old.pdf"
            old_file_path.write_bytes(old_bytes)
            SessionManager.set(
                "pdf_file_path", str(old_file_path), session_id="test_session"
            )

            uploaded = io.BytesIO(new_bytes)
            uploaded.name = "test.pdf"
            uploaded.type = "application/pdf"
            uploaded.size = len(new_bytes)

            fake_state = FakeSessionState({"pdf_uploader": uploaded})
            with (
                patch("main.FilePathConstants.TEMP_DIR", temp_dir),
                patch("main.st.session_state", fake_state),
                patch(
                    "infra.notification_system.SystemNotifier.success"
                ) as mock_success,
            ):
                on_file_upload()

            assert SessionManager.get("new_file_uploaded", session_id="test_session")
            assert SessionManager.get(
                "file_hash", session_id="test_session"
            ) == compute_file_hash("", data=new_bytes)
            assert (
                SessionManager.get("last_uploaded_file_name", session_id="test_session")
                == "test.pdf"
            )
            assert not old_file_path.exists()
            mock_success.assert_called_once()

    def test_on_file_upload_same_name_same_hash_skips_rebuild(self):
        file_bytes = _make_test_pdf("same")
        content_hash = compute_file_hash("", data=file_bytes)

        SessionManager.set(
            "last_uploaded_file_name", "test.pdf", session_id="test_session"
        )
        SessionManager.set("file_hash", content_hash, session_id="test_session")

        uploaded = io.BytesIO(file_bytes)
        uploaded.name = "test.pdf"
        uploaded.type = "application/pdf"
        uploaded.size = len(file_bytes)

        fake_state = FakeSessionState({"pdf_uploader": uploaded})
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("main.FilePathConstants.TEMP_DIR", temp_dir),
            patch("main.st.session_state", fake_state),
            patch("infra.notification_system.SystemNotifier.success") as mock_success,
        ):
            on_file_upload()

        assert not SessionManager.get("new_file_uploaded", session_id="test_session")
        mock_success.assert_called_once()
        assert "already uploaded" in mock_success.call_args.args[0]

    def test_on_file_upload_rejects_corrupt_pdf_without_mutation(self):
        garbage_bytes = b"not a pdf"
        sentinel_path = "sentinel_unreplaced.pdf"

        SessionManager.set("pdf_file_path", sentinel_path, session_id="test_session")

        uploaded = io.BytesIO(garbage_bytes)
        uploaded.name = "test.pdf"
        uploaded.type = "application/pdf"
        uploaded.size = len(garbage_bytes)

        fake_state = FakeSessionState({"pdf_uploader": uploaded})
        with (
            patch("main.st.session_state", fake_state),
            patch("infra.notification_system.SystemNotifier.success") as mock_success,
        ):
            on_file_upload()

        # 검증 실패는 st.error 대신 타임라인 메시지로 기록되어야 한다 (레이아웃 유지).
        messages = SessionManager.get_messages(session_id="test_session")
        assert any("corrupted or unreadable" in m["content"] for m in messages)
        assert not SessionManager.get("new_file_uploaded", session_id="test_session")
        assert (
            SessionManager.get("pdf_file_path", session_id="test_session")
            == sentinel_path
        )
        mock_success.assert_not_called()

    def test_stream_chunks_times_out_on_hanging_rag_stream(self):
        async def never_returning_astream(*args, **kwargs):
            await asyncio.sleep(0.1)

            async def _stream():
                if False:
                    yield

            return _stream()

        async def empty_stream(*args, **kwargs):
            if False:
                yield

        mock_handler = MagicMock()
        mock_handler.stream_graph_events.side_effect = lambda gen: empty_stream()

        with (
            patch(
                "core.rag_core.RAGSystem.astream",
                side_effect=never_returning_astream,
            ),
            patch(
                "ui.components.streaming.get_streaming_handler",
                return_value=mock_handler,
            ),
            patch("ui.components.streaming.UI_STREAMING_TIMEOUT", 0.01),
            pytest.raises(TimeoutError),
        ):
            list(
                stream_chunks(
                    "test query",
                    "test-model",
                    "test_session",
                )
            )

    def test_reset_for_new_file_clears_rebuild_state(self):
        SessionManager.set("pdf_processed", True, session_id="test_session")
        SessionManager.set("rag_engine", object(), session_id="test_session")
        SessionManager.set("rebuild_done", True, session_id="test_session")
        SessionManager.set("rebuild_cancelled", True, session_id="test_session")
        SessionManager.set("needs_rag_rebuild", True, session_id="test_session")
        SessionManager.set("needs_qa_chain_update", True, session_id="test_session")

        SessionManager.reset_for_new_file(session_id="test_session")

        assert not SessionManager.get("pdf_processed", session_id="test_session")
        assert SessionManager.get("rag_engine", session_id="test_session") is None
        assert not SessionManager.get("rebuild_done", session_id="test_session")
        assert not SessionManager.get("rebuild_cancelled", session_id="test_session")
        assert not SessionManager.get("needs_rag_rebuild", session_id="test_session")
        assert not SessionManager.get(
            "needs_qa_chain_update", session_id="test_session"
        )

    def test_is_ready_for_chat_blocks_during_pending_rag_rebuild(self):
        SessionManager.set("pdf_processed", True, session_id="test_session")
        SessionManager.set("rag_engine", object(), session_id="test_session")
        SessionManager.set("needs_rag_rebuild", True, session_id="test_session")

        assert not SessionManager.is_ready_for_chat(session_id="test_session")

        SessionManager.set("needs_rag_rebuild", False, session_id="test_session")
        SessionManager.set("is_building_rag", True, session_id="test_session")

        assert not SessionManager.is_ready_for_chat(session_id="test_session")

    @patch("infra.notification_system.SystemNotifier.error")
    @patch("core.model_loader.load_llm")
    def test_update_qa_chain_exception_handling(
        self, mock_load_llm, mock_notifier_error
    ):
        """QA 체인 업데이트 중 예외 발생 시 세션 상태에 에러가 기록되는지 검증"""

        # 1. 환경 설정
        mock_load_llm.side_effect = Exception("LLM Load Failed Mock Error")
        SessionManager.set("last_selected_model", "test-model")

        # 2. 실행
        _update_qa_chain(session_id="test_session")

        # 3. 검증
        messages = SessionManager.get_messages(session_id="test_session")
        # 어시스턴트 역할로 에러 메시지가 추가되어야 함 (현재 구현 기준)
        assert any("LLM Load Failed Mock Error" in m["content"] for m in messages)
