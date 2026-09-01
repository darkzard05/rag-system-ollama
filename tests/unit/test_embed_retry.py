"""임베딩 콜드스타트 재시도(retry with backoff) 단위 테스트.

계획 §5-3: _NoTruncateOllamaEmbeddings.embed_documents 의 재시도 로직 검증.
"""

from __future__ import annotations

import os
import unittest.mock as mock

import pytest

from core.model_loader import (
    _EMBED_RETRY_BACKOFF_SECONDS,
    _EMBED_RETRY_MAX_ATTEMPTS,
    _is_embed_transient_failure,
)

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _transient_exc(msg: str = "actively refused it.") -> Exception:
    """Ollama 콜드스타트 연결 거부 오류 모의."""
    return Exception(msg)


def _non_retryable_exc() -> Exception:
    """재시도 대상이 아닌 오류 모의."""
    return Exception("model 'missing-model' not found")


def _make_real_embedder():
    """load_embedding_model 을 통해 실제 _NoTruncateOllamaEmbeddings 인스턴스 획득.

    Ollama 미실행 환경에서도 동작하도록 SessionManager.set 만 모의하고,
    생성된 인스턴스에 mock _client 를 주입한다.
    """
    with (
        mock.patch.dict(
            os.environ,
            {"IS_UNIT_TEST": "false", "IS_CI_TEST": "false"},
            clear=False,
        ),
        mock.patch("core.session.SessionManager.set"),
    ):
        from core.model_loader import load_embedding_model

        inst = load_embedding_model("nomic-embed-text")
    # _client 주입 (실제 Ollama 연결 없이)
    inst._client = mock.MagicMock()
    return inst


# ---------------------------------------------------------------------------
# _is_embed_transient_failure 단위 테스트
# ---------------------------------------------------------------------------


class TestIsEmbedTransientFailure:
    """_is_embed_transient_failure 가 연결 거부형 오류를 정확히 판별하는지 검증."""

    @pytest.mark.parametrize(
        "msg",
        [
            "actively refused it.",
            "connection refused",
            "connectex: No connection",
            "actively refused + connectex mixed",
        ],
    )
    def test_transient_markers_detected(self, msg: str) -> None:
        assert _is_embed_transient_failure(Exception(msg)) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "model 'qwen3' not found",
            " unauthorized",
            "timeout",
            "",
        ],
    )
    def test_non_transient_markers_rejected(self, msg: str) -> None:
        assert _is_embed_transient_failure(Exception(msg)) is False


# ---------------------------------------------------------------------------
# embed_documents 재시도 동작 테스트
# ---------------------------------------------------------------------------


class TestEmbedDocumentsRetry:
    """_NoTruncateOllamaEmbeddings.embed_documents 재시도 래퍼 검증."""

    def test_success_on_first_attempt_no_retry(self) -> None:
        """1회 시도 즉시 성공 → 재시도 발동 없음."""
        inst = _make_real_embedder()
        expected = [[0.1, 0.2]]
        inst._client.embed.return_value = {"embeddings": expected}

        result = inst.embed_documents(["hello"])

        assert result == expected
        assert inst._client.embed.call_count == 1

    def test_success_after_two_failures(self) -> None:
        """2회 실패(color refused) 후 3회차 성공 → 최종 성공."""
        inst = _make_real_embedder()
        expected = [[0.3, 0.4]]
        inst._client.embed.side_effect = [
            _transient_exc(),
            _transient_exc(),
            {"embeddings": expected},
        ]

        with mock.patch("time.sleep"):  # 백오프 대기 skip
            result = inst.embed_documents(["hello"])

        assert result == expected
        assert inst._client.embed.call_count == 3

    def test_all_attempts_fail_raises_original_exception(self) -> None:
        """3회 모두 연결 거부 → 원본 예외 재발생."""
        inst = _make_real_embedder()
        original = _transient_exc("actively refused it.")
        inst._client.embed.side_effect = original

        with (
            mock.patch("time.sleep"),
            pytest.raises(Exception, match="actively refused"),
        ):
            inst.embed_documents(["hello"])

        assert inst._client.embed.call_count == _EMBED_RETRY_MAX_ATTEMPTS

    def test_non_retryable_exception_propagates_immediately(self) -> None:
        """재시도 대상이 아닌 예외(model not found 등) → 즉시 전파, 재시도 안 함."""
        inst = _make_real_embedder()
        inst._client.embed.side_effect = _non_retryable_exc()

        with pytest.raises(Exception, match="not found"):
            inst.embed_documents(["hello"])

        assert inst._client.embed.call_count == 1

    def test_retry_logging_on_transient_failure(self) -> None:
        """재시도 발동 시 logger.info 가 호출되는지 검증."""
        inst = _make_real_embedder()
        inst._client.embed.side_effect = [
            _transient_exc(),
            {"embeddings": [[0.5]]},
        ]

        with (
            mock.patch("time.sleep"),
            mock.patch("core.model_loader.logger") as mock_logger,
        ):
            inst.embed_documents(["hello"])

        # 첫 실패 후 재시도 로그 1건 확인
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert "임베딩 콜드스타트 감지" in log_msg

    def test_backoff_wait_seconds_respected(self) -> None:
        """백오프 대기 시간이 정책대로 time.sleep 에 전달되는지 검증."""
        inst = _make_real_embedder()
        inst._client.embed.side_effect = [
            _transient_exc(),
            _transient_exc(),
            {"embeddings": [[0.6]]},
        ]

        with mock.patch("time.sleep") as mock_sleep:
            inst.embed_documents(["hello"])

        # 1차 실패 → 1s, 2차 실패 → 3s
        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == _EMBED_RETRY_BACKOFF_SECONDS[0]
        assert calls[1] == _EMBED_RETRY_BACKOFF_SECONDS[1]

    def test_no_client_raises_value_error(self) -> None:
        """_client 가 None 일 때 ValueError 발생 (재시도 없음)."""
        inst = _make_real_embedder()
        inst._client = None

        with pytest.raises(ValueError, match="not initialized"):
            inst.embed_documents(["hello"])
