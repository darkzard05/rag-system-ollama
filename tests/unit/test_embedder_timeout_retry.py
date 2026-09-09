"""임베딩 타임아웃·재시도·페이로드 분할·배치 크기 해석 단위 테스트 (계획 B11).

- _NoTruncateOllamaEmbeddings: 페이로드 크기 선분할, 비동기 시도별 wait_for
  타임아웃 재시도, 비일시 오류 즉시 전파.
- _resolve_embedding_batch_size: config(EMBEDDING_BATCH_SIZE) 해석 규칙.
"""

from __future__ import annotations

import asyncio
import os
import unittest.mock as mock

import pytest

from core.model_loader import (
    _EMBED_PAYLOAD_CAP_CHARS,
    _EMBED_RETRY_MAX_ATTEMPTS,
    _resolve_embedding_batch_size,
    _split_by_payload_size,
)

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


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
    # _client 주입 (실제 Ollama 연결 없이) — 래퍼(_memo_wrap)가 inst가 되므로 내부로 주입
    inst._inner._client = mock.MagicMock()
    return inst


def _patch_asyncio_sleep() -> mock._patch:
    """백오프 대기(await asyncio.sleep)를 즉시 완료로 대체한다."""
    return mock.patch("asyncio.sleep", new=mock.AsyncMock())


# ---------------------------------------------------------------------------
# _split_by_payload_size 단위 테스트
# ---------------------------------------------------------------------------


class TestSplitByPayloadSize:
    def test_greedy_groups_keep_order(self) -> None:
        texts = ["x" * 3000] * 8  # 총 24_000자 > 12_000 상한

        groups = _split_by_payload_size(texts)

        assert groups == [texts[:4], texts[4:]]
        assert all(sum(len(t) for t in g) <= _EMBED_PAYLOAD_CAP_CHARS for g in groups)

    def test_single_over_cap_text_returned_alone(self) -> None:
        big = "y" * 30_000

        groups = _split_by_payload_size([big, "small"])

        assert groups == [[big], ["small"]]

    def test_empty_input_returns_empty(self) -> None:
        assert _split_by_payload_size([]) == []


# ---------------------------------------------------------------------------
# 페이로드 크기 선분할 동작 (동기 embed_documents)
# ---------------------------------------------------------------------------


class TestEmbedSplitsByPayloadCap:
    def test_embed_splits_by_payload_cap(self) -> None:
        """총량이 상한을 넘으면 그룹으로 나누고 원본 순서를 보존한다."""
        inst = _make_real_embedder()
        texts = ["x" * 3000] * 8  # 24_000자 → 12_000자 × 2 그룹
        offset = 0
        groups_seen: list[list[str]] = []

        def _fake_embed(model, texts, **kwargs):
            nonlocal offset
            groups_seen.append(list(texts))
            vecs = [[float(offset + j), 0.0] for j in range(len(texts))]
            offset += len(texts)
            return {"embeddings": vecs}

        inst._inner._client.embed.side_effect = _fake_embed

        result = inst.embed_documents(texts)

        assert inst._inner._client.embed.call_count == 2
        assert sum(len(t) for t in groups_seen[0]) <= _EMBED_PAYLOAD_CAP_CHARS
        assert sum(len(t) for t in groups_seen[1]) <= _EMBED_PAYLOAD_CAP_CHARS
        # 원본 순서 보존 (그룹 경계를 넘어 0..7 순차 인덱스)
        assert [v[0] for v in result] == [float(i) for i in range(8)]


# ---------------------------------------------------------------------------
# 비동기 타임아웃 재시도 (aembed_documents)
# ---------------------------------------------------------------------------


class TestEmbedAsyncTimeoutRetry:
    async def test_embed_retries_on_timeout(self) -> None:
        """TimeoutError 2회 → 3회차 성공 → 최종 결과 반환."""
        inst = _make_real_embedder()
        expected = [[0.7, 0.8]]
        inst._inner._client.embed.side_effect = [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            {"embeddings": expected},
        ]

        with _patch_asyncio_sleep():
            result = await inst._inner.aembed_documents(["hello"])

        assert result == expected
        assert inst._inner._client.embed.call_count == 3

    async def test_embed_raises_after_max_timeouts(self) -> None:
        """항상 TimeoutError → 최대 시도 후 asyncio.TimeoutError 재발생."""
        inst = _make_real_embedder()
        inst._inner._client.embed.side_effect = asyncio.TimeoutError()

        with (
            _patch_asyncio_sleep(),
            pytest.raises(asyncio.TimeoutError),
        ):
            await inst._inner.aembed_documents(["hello"])

        assert inst._inner._client.embed.call_count == _EMBED_RETRY_MAX_ATTEMPTS

    async def test_non_transient_embed_error_no_retry(self) -> None:
        """비일시 오류(ValueError) → 즉시 전파, 단 1회 시도."""
        inst = _make_real_embedder()
        inst._inner._client.embed.side_effect = ValueError("model 'ghost' not found")

        with pytest.raises(ValueError, match="not found"):
            await inst._inner.aembed_documents(["hello"])

        assert inst._inner._client.embed.call_count == 1

    async def test_aembed_query_single_text_uses_shared_retry(self) -> None:
        """aembed_query 는 단일 텍스트로 공용 재시도 루프를 거친다."""
        inst = _make_real_embedder()
        inst._inner._client.embed.side_effect = [
            asyncio.TimeoutError(),
            {"embeddings": [[0.1, 0.2]]},
        ]

        with _patch_asyncio_sleep():
            result = await inst._inner.aembed_query("hello")

        assert result == [0.1, 0.2]
        assert inst._inner._client.embed.call_count == 2


# ---------------------------------------------------------------------------
# batch_size 해석 (_resolve_embedding_batch_size)
# ---------------------------------------------------------------------------


class TestResolveEmbeddingBatchSize:
    def test_auto_cuda_and_cpu_defaults(self) -> None:
        with mock.patch("core.model_loader.EMBEDDING_BATCH_SIZE", "auto"):
            assert _resolve_embedding_batch_size("cuda") == 32
            assert _resolve_embedding_batch_size("cpu") == 16

    def test_int_used_as_is(self) -> None:
        with mock.patch("core.model_loader.EMBEDDING_BATCH_SIZE", 8):
            assert _resolve_embedding_batch_size("cuda") == 8
            assert _resolve_embedding_batch_size("cpu") == 8

    def test_numeric_string_parsed(self) -> None:
        with mock.patch("core.model_loader.EMBEDDING_BATCH_SIZE", "8"):
            assert _resolve_embedding_batch_size("cpu") == 8

    @pytest.mark.parametrize("invalid", ["lots", 0, -3, True, 3.5])
    def test_invalid_falls_back_to_16(self, invalid: object) -> None:
        with (
            mock.patch("core.model_loader.EMBEDDING_BATCH_SIZE", invalid),
            mock.patch("core.model_loader.logger") as mock_logger,
        ):
            assert _resolve_embedding_batch_size("cuda") == 16
        mock_logger.warning.assert_called()
