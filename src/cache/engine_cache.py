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

        if not rag_engine or cached_loop_id != current_loop_id:
            if rag_engine:
                logger.info(
                    f"[RAG] [ENGINE] 루프 변경({cached_loop_id} -> {current_loop_id}): 캐시 무효화"
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

        SessionManager.set("rag_engine", engine, session_id=session_id)
        SessionManager.set("rag_engine_loop_id", current_loop_id, session_id=session_id)
        logger.info(f"[RAG] [ENGINE] 엔진 캐시됨 (loop_id={current_loop_id})")
