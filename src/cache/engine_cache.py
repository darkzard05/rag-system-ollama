from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.session import SessionManager

logger = logging.getLogger(__name__)


class EngineCacheManager:
    @staticmethod
    def get_engine(session_id: str) -> Any | None:
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        rag_engine = SessionManager.get("rag_engine", session_id=session_id)
        cached_loop_id = SessionManager.get(
            "rag_engine_loop_id", 0, session_id=session_id
        )
        cached_file_hash = SessionManager.get(
            "rag_engine_file_hash", session_id=session_id
        )
        current_file_hash = SessionManager.get("file_hash", session_id=session_id)

        if (
            not rag_engine
            or cached_loop_id != current_loop_id
            or cached_file_hash != current_file_hash
        ):
            if rag_engine:
                logger.info(
                    "[RAG] [ENGINE] 캐시 무효화 "
                    f"(loop: {cached_loop_id}->{current_loop_id}, "
                    f"file_hash: {cached_file_hash!r}->{current_file_hash!r})"
                )
            return None

        logger.info("[RAG] [ENGINE] 캐시된 rag_engine 사용")
        return rag_engine

    @staticmethod
    def set_engine(session_id: str, engine: Any) -> None:
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        # 엔진이 참조하는 문서의 해시를 함께 기록해, 이후 file_hash가 바뀌면
        # get_engine이 이전 문서 기준 엔진을 반환하지 않게 합니다 (팬텀 상태 방지).
        file_hash = SessionManager.get("file_hash", session_id=session_id)
        SessionManager.set("rag_engine", engine, session_id=session_id)
        SessionManager.set("rag_engine_loop_id", current_loop_id, session_id=session_id)
        SessionManager.set("rag_engine_file_hash", file_hash, session_id=session_id)
        logger.info(f"[RAG] [ENGINE] 엔진 캐시됨 (loop_id={current_loop_id})")
