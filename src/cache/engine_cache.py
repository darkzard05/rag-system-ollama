from __future__ import annotations

import logging
from typing import Any

from core.session import SessionManager

logger = logging.getLogger(__name__)

# SessionManager 가 엔진의 단일 소스(SSoT)이자 수명주기를 관리한다 (스레드 안전,
# 세션 생성/삭제 시 정리). 문서 해시(file_hash)가 유효성 키이며, 이벤트 루프 id 는
# 쓰지 않는다 — set/get 이 서로 다른 이벤트 루프(업로드 빌드=AsyncWorker 루프,
# 쿼리 스트림=`asyncio.run` 새 루프)에서 실행되므로 루프 id 로 묶으면 항상 무효화된다.


class EngineCacheManager:
    @staticmethod
    def get_engine(session_id: str) -> Any | None:
        rag_engine = SessionManager.get("rag_engine", session_id=session_id)
        cached_file_hash = SessionManager.get(
            "rag_engine_file_hash", session_id=session_id
        )
        current_file_hash = SessionManager.get("file_hash", session_id=session_id)

        if not rag_engine or cached_file_hash != current_file_hash:
            if rag_engine:
                logger.info(
                    "[RAG] [ENGINE] 캐시 무효화 "
                    f"(file_hash: {cached_file_hash!r}->{current_file_hash!r})"
                )
            return None

        logger.info("[RAG] [ENGINE] 캐시된 rag_engine 사용")
        return rag_engine

    @staticmethod
    def set_engine(session_id: str, engine: Any) -> None:
        # 엔진이 참조하는 문서의 해시를 함께 기록해, 이후 file_hash가 바뀌면
        # get_engine이 이전 문서 기준 엔진을 반환하지 않게 합니다 (팬텀 상태 방지).
        file_hash = SessionManager.get("file_hash", session_id=session_id)
        SessionManager.set("rag_engine", engine, session_id=session_id)
        SessionManager.set("rag_engine_file_hash", file_hash, session_id=session_id)
        logger.info("[RAG] [ENGINE] 엔진 캐시됨")
