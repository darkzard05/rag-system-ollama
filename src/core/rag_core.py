"""
RAG 시스템의 통합 엔트리포인트.
기능별로 분리된 모듈(document_processor, retriever_manager)을 조정합니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.embeddings import Embeddings

from common.exceptions import (
    VectorStoreError,
)
from common.typing_utils import T
from core.retriever_manager import build_rag_pipeline
from core.session import SessionManager

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG 시스템의 통합 엔트리포인트 클래스.
    세션 기반 상태 관리와 LangGraph 기반 파이프라인을 연결합니다.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        SessionManager.init_session(session_id=session_id)

    def _ensure_session_context(self) -> None:
        """현재 스레드의 세션 컨텍스트를 보장합니다."""
        SessionManager.set_session_id(self.session_id)

    async def load_document(
        self, file_path: str, file_name: str, embedder: Embeddings
    ) -> tuple[str, bool]:
        """
        문서를 로드하고 인덱싱 파이프라인을 실행합니다.
        """
        self._ensure_session_context()
        return await asyncio.to_thread(
            build_rag_pipeline,
            uploaded_file_name=file_name,
            file_path=file_path,
            embedder=embedder,
            session_id=self.session_id,
        )

    async def aquery(self, query: str, llm: T | None = None) -> dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.
        """
        self._ensure_session_context()

        if llm:
            SessionManager.set("llm", llm, session_id=self.session_id)

        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={
                    "reason": "RAG 엔진이 초기화되지 않았습니다. 문서를 먼저 로드하세요."
                }
            )

        current_llm = SessionManager.get("llm", session_id=self.session_id)

        # [최적화] 세션별 리소스를 Config에 주입하여 그래프 노드에서의 격리성 보장
        config = {
            "configurable": {
                "llm": current_llm,
                "session_id": self.session_id,
                "thread_id": self.session_id,
                "faiss_retriever": SessionManager.get(
                    "faiss_retriever", session_id=self.session_id
                ),
                "bm25_retriever": SessionManager.get(
                    "bm25_retriever", session_id=self.session_id
                ),
                "doc_language": SessionManager.get(
                    "doc_language", session_id=self.session_id
                ),
            }
        }

        # LangGraph 호출
        return await rag_engine.ainvoke({"input": query}, config=config)

    def get_status(self) -> list[str]:
        """현재 세션의 작업 로그를 가져옵니다."""
        self._ensure_session_context()
        return SessionManager.get("status_logs", session_id=self.session_id) or []

    def clear_session(self) -> None:
        """세션 데이터를 초기화합니다."""
        self._ensure_session_context()
        SessionManager.reset_all_state(session_id=self.session_id)
