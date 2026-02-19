"""
LLM 성능 추적 및 로그 기록을 담당하는 모듈입니다.
"""

import contextlib
import logging
import time
from typing import Any

from services.monitoring.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


class ResponsePerformanceTracker:
    def __init__(self, query_text: str, model_instance: Any):
        # 지연 임포트로 순환 참조 방지
        from core.session import SessionManager

        self.SessionManager = SessionManager
        self.start_time: float = time.time()
        self.query: str = query_text
        self.model: Any = model_instance
        self.first_token_at: float | None = None
        self.thinking_started_at: float | None = None
        self.thinking_finished_at: float | None = None
        self.answer_started_at: float | None = None
        self.answer_finished_at: float | None = None
        self.chunk_count: int = 0
        self._resp_parts: list[str] = []
        self._thought_parts: list[str] = []
        self.full_response: str = ""
        self.full_thought: str = ""
        self.context: str = ""
        self.input_token_count: int = 0
        self.doc_count: int = 0
        self._log_thinking_start: bool = False
        self._log_answer_start: bool = False

    def set_context(self, context_text: str, doc_count: int = 0):
        """평가를 위해 사용된 컨텍스트와 문서 개수를 기록합니다."""
        from common.utils import count_tokens_rough

        self.context = context_text
        self.doc_count = doc_count
        # 질문 + 컨텍스트 합산 입력 토큰 계산
        self.input_token_count = count_tokens_rough(self.query + context_text)

    def record_chunk(self, content: str, thought: str):
        now = time.time()
        if self.first_token_at is None:
            self.first_token_at = now

        if thought:
            if not self._log_thinking_start:
                self.SessionManager.replace_last_status_log("사고 과정 기록 중...")
                self.thinking_started_at = now
                self._log_thinking_start = True
            self._thought_parts.append(thought)

        if content:
            # 중복 기록 방지
            if (
                not self._resp_parts or self._resp_parts[-1] != content
            ) and not self._log_answer_start:
                if self.thinking_started_at and self.thinking_finished_at is None:
                    self.thinking_finished_at = now

                self.SessionManager.replace_last_status_log("답변 스트리밍 중...")
                self.answer_started_at = now
                self._log_answer_start = True

            if not self._resp_parts or self._resp_parts[-1] != content:
                self._resp_parts.append(content)
                self.chunk_count += 1

    def finalize_and_log(self) -> Any:
        from api.schemas import PerformanceStats
        from common.utils import count_tokens_rough

        self.answer_finished_at = time.time()
        self.full_response = "".join(self._resp_parts)
        self.full_thought = "".join(self._thought_parts)

        total_duration = self.answer_finished_at - self.start_time
        time_to_first_token = (
            (self.first_token_at - self.start_time) if self.first_token_at else 0
        )

        thinking_duration: float = 0.0
        if self.thinking_started_at:
            end_time = (
                self.thinking_finished_at
                or self.answer_started_at
                or self.answer_finished_at
            )
            thinking_duration = end_time - self.thinking_started_at

        answer_duration: float = (
            (self.answer_finished_at - self.answer_started_at)
            if self.answer_started_at
            else 0.0
        )

        resp_token_count = count_tokens_rough(self.full_response)
        thought_token_count = count_tokens_rough(self.full_thought)

        stats = PerformanceStats(
            ttft=time_to_first_token,
            thinking_time=thinking_duration,
            generation_time=answer_duration,
            total_time=total_duration,
            token_count=resp_token_count,
            thought_token_count=thought_token_count,
            input_token_count=self.input_token_count,
            doc_count=self.doc_count,
            model_name=getattr(self.model, "model", "unknown"),
        )

        # [로그 개선] 모든 지표를 한 줄에 가독성 있게 출력
        logger.info(
            f"[LLM] 완료 | "
            f"TTFT: {stats.ttft:.2f}s | "
            f"Tokens: {stats.input_token_count} (In) / {stats.token_count} (Out) / {stats.thought_token_count} (Thought) | "
            f"Speed: {stats.tps:.1f} tok/s | "
            f"Docs: {stats.doc_count} chunks | "
            f"Total: {stats.total_time:.2f}s"
        )

        self.SessionManager.replace_last_status_log(
            f"완료 (In {stats.input_token_count} / Out {stats.token_count} / Thought {stats.thought_token_count})"
        )

        with contextlib.suppress(Exception):
            # 1. 성능 메트릭 (CSV)
            monitor.log_to_csv(
                {
                    "model": stats.model_name,
                    "ttft": stats.ttft,
                    "thinking": stats.thinking_time,
                    "answer": stats.generation_time,
                    "total": stats.total_time,
                    "tokens_in": stats.input_token_count,
                    "tokens_out": stats.token_count,
                    "tokens_thought": stats.thought_token_count,
                    "doc_count": stats.doc_count,
                    "tps": stats.tps,
                    "query": self.query,
                }
            )
            # 2. 통합 히스토리 (JSONL)
            monitor.log_qa_history(
                {
                    "session_id": self.SessionManager.get_session_id(),
                    "query": self.query,
                    "context": self.context,
                    "thought": self.full_thought,
                    "response": self.full_response,
                    "metrics": stats.model_dump(),
                }
            )
        return stats
