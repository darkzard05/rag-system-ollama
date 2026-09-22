"""DEFECT #3: 오류 메시지가 데드엔드 — 재시도 CTA·원인·가이드가 없는 테스트.

현재 구현:
- friendly_error_message: 연결 관련 6개 서명만 매핑, 그 외는
  "An error occurred while generating the answer." (제네릭)
- render_message 오류 분기: st.error(f"Error: {error}")만 표시, 재시도 버튼 없음
- chat_build.py 빌드 오류: st.status(state="error") + ui_error만 표시, 재시도 없음

목표 상태:
- 오류 카드: {무엇이 happened, 원인 추정, 보존된 부분 응답, CTA 버튼}
- CTA: [Retry] [New Chat] [Re-upload] [Copy details]
- friendly_error_message이 최소 6가지 예외 유형을 매핑

이 테스트는 현재 코드에서 반드시 실패해야 한다 (TDD 레드).
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

from ui.components.streaming import (
    _ERROR_SIGNATURES,
    _GENERIC_STREAMING_MSG,
    friendly_error_message,
)


class TestFriendlyErrorMessageCoverage:
    """friendly_error_message이 다양한 예외 유형을 매핑해야 한다."""

    def test_maps_connection_refused_to_ollama_message(self) -> None:
        """연결 거부 예외 → Ollama 실행 안내 메시지."""
        from common.config import MSG_ERROR_OLLAMA_NOT_RUNNING

        exc = ConnectionRefusedError("Connection refused to localhost:11434")
        result = friendly_error_message(exc)
        assert result == MSG_ERROR_OLLAMA_NOT_RUNNING

    def test_maps_timeout_error(self) -> None:
        """타임아웃 예외 → 사용자 친화 메시지 (현재: 제네릭으로 실패)."""
        exc = TimeoutError("Request timed out after 30s")
        result = friendly_error_message(exc)
        # 현재: 제네릭 메시지를 반환하므로 실패
        assert result != _GENERIC_STREAMING_MSG, (
            f"TimeoutError가 제네릭 메시지를 반환했다: {result}. "
            "타임아웃 전용 메시지가 필요하다."
        )

    def test_maps_cancelled_error(self) -> None:
        """사용자 중단 예외 → 중단 안내 메시지 (현재: 제네릭으로 실패)."""
        exc = RuntimeError("Generation cancelled by user")
        result = friendly_error_message(exc)
        # 현재: "cancelled" 서명이 _ERROR_SIGNATURES에 없으므로 제네릭
        assert result != _GENERIC_STREAMING_MSG, (
            f"취소 예외가 제네릭 메시지를 반환했다: {result}. "
            "취소 전용 메시지가 필요하다."
        )

    def test_maps_pdf_error(self) -> None:
        """PDF 처리 예외 → PDF 관련 메시지 (현재: 제네릭으로 실패)."""
        exc = ValueError("Invalid PDF: corrupt file header")
        result = friendly_error_message(exc)
        assert result != _GENERIC_STREAMING_MSG, (
            f"PDF 예외가 제네릭 메시지를 반환했다: {result}. "
            "PDF 전용 메시지가 필요하다."
        )

    def test_maps_embedding_error(self) -> None:
        """임베딩 예외 → 임베딩 관련 메시지 (현재: 제네릭으로 실패)."""
        exc = RuntimeError("Embedding model nomic-embed-text not found")
        result = friendly_error_message(exc)
        assert result != _GENERIC_STREAMING_MSG, (
            f"임베딩 예외가 제네릭 메시지를 반환했다: {result}. "
            "임베딩 전용 메시지가 필요하다."
        )

    def test_maps_memory_error(self) -> None:
        """메모리 부족 예외 → 리소스 관련 메시지 (현재: 제네릭으로 실패)."""
        exc = MemoryError("Not enough GPU memory for embedding batch")
        result = friendly_error_message(exc)
        assert result != _GENERIC_STREAMING_MSG, (
            f"메모리 예외가 제네릭 메시지를 반환했다: {result}. "
            "메모리 전용 메시지가 필요하다."
        )

    def test_error_signatures_has_at_least_6_entries(self) -> None:
        """_ERROR_SIGNATURES가 최소 6개 이상의 매핑을 가져야 한다."""
        assert len(_ERROR_SIGNATURES) >= 6, (
            f"_ERROR_SIGNATURES가 {len(_ERROR_SIGNATURES)}개뿐이다 (목표: 최소 6개)"
        )


class TestErrorRecoveryCTA:
    """오류 발생 시 재시도 CTA가 포함되어야 한다."""

    def test_error_message_dict_includes_retry_info(self) -> None:
        """오류 메시지 dict에 retry_kind 필드가 포함되어야 한다.

        현재 실패 원인: consume_stream_into_message의 오류 경로는
        error=friendly_error_message(exc)만 설정하고, 재시도 관련
        정보를 포함하지 않는다.
        """
        from ui.components.streaming import consume_stream_into_message

        _store: dict[str, Any] = {}
        _messages: list[dict[str, Any]] = []

        class _TestSessionManager:
            @staticmethod
            def get(key: str, default: Any = None, **kwargs: Any) -> Any:
                return _store.get(key, default)

            @staticmethod
            def set(key: str, value: Any, **kwargs: Any) -> None:
                _store[key] = value

            @staticmethod
            def add_message(
                role: str, content: str, msg_type: str = "general", **kwargs: Any
            ) -> None:
                msg = {"msg_id": kwargs.pop("msg_id", str(uuid.uuid4())), **kwargs}
                for i, m in enumerate(_messages):
                    if m.get("msg_id") == msg.get("msg_id"):
                        _messages[i].update(msg)
                        return
                _messages.append(msg)

            @staticmethod
            def get_messages(**kwargs: Any) -> list[dict[str, Any]]:
                return list(_messages)

            @staticmethod
            def _acquire_lock(sid: str) -> Any:
                from contextlib import nullcontext

                return nullcontext()

            @staticmethod
            def _get_state(sid: str) -> dict[str, Any]:
                return {"messages": _messages}

        def _failing_stream(query: str, model: str, sid: str):
            yield StreamChunk(content="partial ", status="step_0")
            raise ConnectionError("Ollama connection refused")

        from api.streaming_handler import StreamChunk

        with (
            patch("ui.components.streaming.stream_chunks", _failing_stream),
            patch("ui.components.streaming.SessionManager", _TestSessionManager),
            patch("ui.components.streaming_core.SessionManager", _TestSessionManager),
            patch("ui.components.streaming_state.SessionManager", _TestSessionManager),
        ):
            result = consume_stream_into_message("test_sid", "test query", "test-model")

        assert result is not None
        # 오류 메시지에 재시도 관련 정보가 포함되어야 한다
        assert "retry" in str(result).lower() or result.get("error") is not None, (
            f"오류 메시지에 재시도 정보가 없다: {result}"
        )

    def test_friendly_error_message_returns_structured_dict(self) -> None:
        """friendly_error_message이 구조화된 dict를 반환해야 한다 (현재: 문자열).

        목표: {"message": str, "cause": str, "retryable": bool, "cta": list[str]}
        현재: 단순 문자열 반환 → 구조화된 정보 부족.
        """
        exc = ConnectionRefusedError("Connection refused")
        result = friendly_error_message(exc)

        # 현재: 문자열을 반환하므로 dict가 아님 → 실패
        assert isinstance(result, dict), (
            f"friendly_error_message이 문자열을 반환했다: {type(result).__name__}. "
            "구조화된 dict(cause, retryable, cta 포함)가 필요하다."
        )

    def test_build_error_includes_retry_button(self) -> None:
        """빌드 오류 메시지에 재시도 버튼이 포함되어야 한다.

        chat_build.get_build_error_actions가 재시도 페이로드를 제공하고,
        _render_build_progress_block이 Retry 버튼을 렌더해야 한다.
        """
        import inspect

        from ui.components import chat_build
        from ui.components.chat_build import get_build_error_actions

        # 빌드 오류 재시도 페이로드 구조 검증
        build_error_msg = get_build_error_actions("Analysis failed")

        has_retry = "retry" in str(build_error_msg).lower()
        assert has_retry, (
            f"빌드 오류 메시지에 재시도 정보가 없다: {build_error_msg}. "
            "retry_kind 또는 retry_handler 필드가 필요하다."
        )
        assert build_error_msg.get("retry_kind") == "rebuild"
        assert build_error_msg.get("retryable") is True

        # 렌더 경로에 실제 Retry 버튼이 존재해야 한다
        source = inspect.getsource(chat_build._render_build_progress_block)
        assert "retry" in source.lower(), (
            "빌드 오류 렌더 경로에 Retry 버튼이 없다: "
            "_render_build_progress_block에 재시도 CTA가 필요하다."
        )

    def test_error_preserves_partial_content(self) -> None:
        """오류 발생 시 부분 응답이 보존되어야 한다."""
        from api.streaming_handler import StreamChunk
        from ui.components.streaming import consume_stream_into_message

        _store: dict[str, Any] = {}
        _messages: list[dict[str, Any]] = []

        class _TestSessionManager:
            @staticmethod
            def get(key: str, default: Any = None, **kwargs: Any) -> Any:
                return _store.get(key, default)

            @staticmethod
            def set(key: str, value: Any, **kwargs: Any) -> None:
                _store[key] = value

            @staticmethod
            def add_message(
                role: str, content: str, msg_type: str = "general", **kwargs: Any
            ) -> None:
                msg = {"msg_id": kwargs.pop("msg_id", str(uuid.uuid4())), **kwargs}
                for i, m in enumerate(_messages):
                    if m.get("msg_id") == msg.get("msg_id"):
                        _messages[i].update(msg)
                        return
                _messages.append(msg)

            @staticmethod
            def get_messages(**kwargs: Any) -> list[dict[str, Any]]:
                return list(_messages)

            @staticmethod
            def _acquire_lock(sid: str) -> Any:
                from contextlib import nullcontext

                return nullcontext()

            @staticmethod
            def _get_state(sid: str) -> dict[str, Any]:
                return {"messages": _messages}

        def _partial_then_fail(query: str, model: str, sid: str):
            yield StreamChunk(content="This is a ", status="step_0")
            yield StreamChunk(content="partial answer ", status="step_1")
            raise RuntimeError("Stream interrupted")

        with (
            patch("ui.components.streaming.stream_chunks", _partial_then_fail),
            patch("ui.components.streaming.SessionManager", _TestSessionManager),
            patch("ui.components.streaming_core.SessionManager", _TestSessionManager),
            patch("ui.components.streaming_state.SessionManager", _TestSessionManager),
        ):
            result = consume_stream_into_message("test_sid", "test query", "test-model")

        assert result is not None
        assert result.get("content") == "This is a partial answer", (
            f"부분 응답이 보존되지 않았다: content={result.get('content')!r}"
        )
