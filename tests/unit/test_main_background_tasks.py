# src/main.py의 백그라운드 스레드 작업 예외 처리를 검증하는 테스트
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath("src"))

from src.core.session import SessionManager


class TestMainBackgroundTasks(unittest.TestCase):
    def setUp(self):
        SessionManager.reset_all_state("test_session")
        SessionManager.set_session_id("test_session")

    @patch("infra.notification_system.SystemNotifier.error")
    @patch("src.core.rag_core.RAGSystem.build_pipeline")
    def test_rebuild_rag_system_exception_handling(
        self, mock_build, mock_notifier_error
    ):
        """RAG 빌드 중 예외 발생 시 세션 상태에 에러가 기록되는지 검증"""
        import asyncio
        from main import _bg_rebuild_task

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
        self.assertTrue(SessionManager.get("pdf_processed", session_id="test_session"))
        error_msg = SessionManager.get("pdf_processing_error", session_id="test_session")
        self.assertIn("RAG Build Failed Mock Error", error_msg)
    
        # 시스템 메시지에 에러가 추가되었는지 확인
        messages = SessionManager.get_messages(session_id="test_session")
        self.assertTrue(
            any("RAG Build Failed Mock Error" in m["content"] for m in messages)
        )

    @patch("infra.notification_system.SystemNotifier.error")
    @patch("src.core.model_loader.load_llm")
    def test_update_qa_chain_exception_handling(
        self, mock_load_llm, mock_notifier_error
    ):
        """QA 체인 업데이트 중 예외 발생 시 세션 상태에 에러가 기록되는지 검증"""
        from main import _update_qa_chain

        # 1. 환경 설정
        mock_load_llm.side_effect = Exception("LLM Load Failed Mock Error")
        SessionManager.set("last_selected_model", "test-model")

        # 2. 실행
        _update_qa_chain(session_id="test_session")

        # 3. 검증
        messages = SessionManager.get_messages(session_id="test_session")
        # 어시스턴트 역할로 에러 메시지가 추가되어야 함 (현재 구현 기준)
        self.assertTrue(
            any("LLM Load Failed Mock Error" in m["content"] for m in messages)
        )


if __name__ == "__main__":
    unittest.main()
