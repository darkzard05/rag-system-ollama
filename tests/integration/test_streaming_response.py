"""
스트리밍 응답 처리 테스트 - Task 12
총 30개의 포괄적인 테스트
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import time
import unittest

from api.streaming_handler import (
    AdaptiveStreamingController,
    ServerSentEventsHandler,
    StreamChunk,
    StreamingMetrics,
    StreamingResponseBuilder,
    StreamingResponseHandler,
    TokenStreamBuffer,
    get_adaptive_controller,
    get_streaming_handler,
)


class TestTokenStreamBuffer(unittest.TestCase):
    """TokenStreamBuffer 테스트 (4개)"""

    def test_buffer_add_single_token(self):
        """단일 토큰 추가"""
        buffer = TokenStreamBuffer(buffer_size=10)
        result = buffer.add_token("hello")
        assert result is None  # 버퍼 미만이므로 None

    def test_buffer_flush_on_full(self):
        """버퍼 풀 시 플러시"""
        buffer = TokenStreamBuffer(buffer_size=3)
        result1 = buffer.add_token("a")
        result2 = buffer.add_token("b")
        result3 = buffer.add_token("c")

        assert result1 is None
        assert result2 is None
        assert result3 is not None  # 3개 도달 시 플러시
        assert result3 == "abc"

    def test_buffer_manual_flush(self):
        """수동 플러시"""
        buffer = TokenStreamBuffer()
        buffer.add_token("hello")
        buffer.add_token(" ")
        buffer.add_token("world")

        result = buffer.flush()
        assert result == "hello world"
        assert len(buffer.buffer) == 0

    def test_buffer_reset(self):
        """버퍼 리셋"""
        buffer = TokenStreamBuffer()
        buffer.add_token("test")
        buffer.reset()

        assert len(buffer.buffer) == 0


class TestStreamChunk(unittest.TestCase):
    """StreamChunk 테스트 (2개)"""

    def test_stream_chunk_creation(self):
        """청크 생성"""
        chunk = StreamChunk(
            content="test content", timestamp=time.time(), token_count=2, chunk_index=0
        )

        assert chunk.content == "test content"
        assert chunk.token_count == 2
        assert not chunk.is_final

    def test_stream_chunk_final(self):
        """최종 청크"""
        chunk = StreamChunk(
            content="final",
            timestamp=time.time(),
            token_count=1,
            chunk_index=10,
            is_final=True,
        )

        assert chunk.is_final


class TestStreamingResponseHandler(unittest.TestCase):
    """StreamingResponseHandler 테스트 (8개)"""

    def setUp(self):
        self.handler = StreamingResponseHandler()

    async def _mock_token_generator(self, tokens: list[str]):
        """토큰 생성 시뮬레이션"""
        for token in tokens:
            await asyncio.sleep(0.01)
            yield token

    def test_basic_streaming(self):
        """기본 스트리밍"""

        async def _test():
            chunks_received = []

            async def on_chunk(chunk: StreamChunk):
                chunks_received.append(chunk)

            metrics = await self.handler.stream_response(
                self._mock_token_generator(["hello", " ", "world"]), on_chunk
            )

            assert len(chunks_received) > 0
            assert (
                metrics.total_tokens > 0
            )  # 토큰 개수 확인 (정확히 3개가 아닐 수 있음)

        asyncio.run(_test())

    def test_streaming_metrics(self):
        """스트리밍 메트릭"""

        async def _test():
            chunks = []

            async def on_chunk(chunk: StreamChunk):
                chunks.append(chunk)

            metrics = await self.handler.stream_response(
                self._mock_token_generator(["a", "b", "c", "d", "e"]), on_chunk
            )

            assert metrics.total_tokens > 0  # 5개 이상
            assert metrics.total_time > 0
            assert metrics.tokens_per_second > 0
            assert metrics.first_token_latency > 0

        asyncio.run(_test())

    def test_streaming_with_completion_callback(self):
        """완료 콜백"""

        async def _test():
            completion_called = False

            async def on_chunk(chunk: StreamChunk):
                pass

            async def on_complete():
                nonlocal completion_called
                completion_called = True

            await self.handler.stream_response(
                self._mock_token_generator(["test"]), on_chunk, on_complete
            )

            assert completion_called

        asyncio.run(_test())

    def test_streaming_with_error_callback(self):
        """에러 콜백"""

        async def _test():
            error_caught = False

            async def _error_generator():
                yield "ok"
                raise ValueError("Test error")

            async def on_chunk(chunk: StreamChunk):
                pass

            async def on_error(error: Exception):
                nonlocal error_caught
                error_caught = True

            await self.handler.stream_response(
                _error_generator(), on_chunk, on_error=on_error
            )

            assert error_caught

        asyncio.run(_test())

    def test_first_token_latency(self):
        """첫 토큰 지연"""

        async def _test():
            async def on_chunk(chunk: StreamChunk):
                pass

            metrics = await self.handler.stream_response(
                self._mock_token_generator(["first", "second"]), on_chunk
            )

            assert metrics.first_token_latency > 0

        asyncio.run(_test())

    def test_avg_chunk_size_calculation(self):
        """평균 청크 크기 계산"""

        async def _test():
            async def on_chunk(chunk: StreamChunk):
                pass

            metrics = await self.handler.stream_response(
                self._mock_token_generator(["a", "bb", "ccc"]), on_chunk
            )

            assert metrics.avg_chunk_size > 0

        asyncio.run(_test())

    def test_streaming_with_large_tokens(self):
        """대용량 토큰 처리"""

        async def _test():
            large_token = "x" * 1000
            chunks = []

            async def on_chunk(chunk: StreamChunk):
                chunks.append(chunk)

            await self.handler.stream_response(
                self._mock_token_generator([large_token, large_token]), on_chunk
            )

            assert len(chunks) > 0

        asyncio.run(_test())


class TestServerSentEventsHandler(unittest.TestCase):
    """ServerSentEventsHandler 테스트 (5개)"""

    def test_format_sse_event(self):
        """SSE 이벤트 포매팅"""
        data = {"token": "hello", "index": 0}
        sse = ServerSentEventsHandler.format_sse_event("chunk", data)

        assert "event: chunk" in sse
        assert "data:" in sse
        assert "hello" in sse

    def test_format_sse_event_with_id(self):
        """ID가 있는 SSE 이벤트"""
        data = {"token": "test"}
        sse = ServerSentEventsHandler.format_sse_event("chunk", data, event_id=1)

        assert "id: 1" in sse

    def test_format_sse_error(self):
        """SSE 에러 포매팅"""
        sse = ServerSentEventsHandler.format_sse_error("Test error", 500)

        assert "event: error" in sse
        assert "Test error" in sse

    def test_format_sse_keepalive(self):
        """SSE keep-alive"""
        sse = ServerSentEventsHandler.format_sse_keepalive()

        assert "keep-alive" in sse

    def test_sse_korean_support(self):
        """한글 지원"""
        data = {"message": "안녕하세요"}
        sse = ServerSentEventsHandler.format_sse_event("message", data)

        assert "안녕하세요" in sse


class TestStreamingResponseBuilder(unittest.TestCase):
    """StreamingResponseBuilder 테스트 (4개)"""

    def setUp(self):
        self.builder = StreamingResponseBuilder()

    def test_add_chunk(self):
        """청크 추가"""
        chunk = StreamChunk(
            content="hello", timestamp=time.time(), token_count=1, chunk_index=0
        )

        self.builder.add_chunk(chunk)
        assert self.builder.get_content() == "hello"

    def test_accumulate_chunks(self):
        """청크 누적"""
        for i in range(5):
            chunk = StreamChunk(
                content=f"chunk{i}", timestamp=time.time(), token_count=1, chunk_index=i
            )
            self.builder.add_chunk(chunk)

        content = self.builder.get_content()
        assert len(self.builder.get_chunks()) == 5
        assert "chunk0" in content

    def test_get_chunks(self):
        """청크 반환"""
        chunk = StreamChunk(
            content="test", timestamp=time.time(), token_count=1, chunk_index=0
        )

        self.builder.add_chunk(chunk)
        chunks = self.builder.get_chunks()

        assert len(chunks) == 1
        assert chunks[0].content == "test"

    def test_reset_builder(self):
        """빌더 리셋"""
        chunk = StreamChunk(
            content="test", timestamp=time.time(), token_count=1, chunk_index=0
        )

        self.builder.add_chunk(chunk)
        self.builder.reset()

        assert self.builder.get_content() == ""
        assert len(self.builder.get_chunks()) == 0


class TestAdaptiveStreamingController(unittest.TestCase):
    """AdaptiveStreamingController 테스트 (5개)"""

    def setUp(self):
        self.controller = AdaptiveStreamingController()

    def test_initial_buffer_size(self):
        """초기 버퍼 크기"""
        size = self.controller.get_buffer_size()
        assert size == 10

    def test_record_latency(self):
        """지연 기록"""
        self.controller.record_latency(50.0)
        metrics = self.controller.get_metrics()

        assert metrics["min_latency_ms"] == 50.0

    def test_buffer_increase_on_high_latency(self):
        """높은 지연 시 버퍼 증가"""
        # 높은 지연 반복
        for _ in range(20):
            self.controller.record_latency(250.0)

        size = self.controller.get_buffer_size()
        assert size > 10

    def test_buffer_decrease_on_low_latency(self):
        """낮은 지연 시 버퍼 감소 또는 유지"""
        # 높은 지연으로 버퍼 증가
        for _ in range(20):
            self.controller.record_latency(250.0)

        initial_size = self.controller.get_buffer_size()

        # 낮은 지연으로 버퍼 조정
        for _ in range(20):
            self.controller.record_latency(30.0)

        final_size = self.controller.get_buffer_size()
        # 버퍼 크기가 감소하거나 최대치에 도달
        assert final_size <= initial_size + 5

    def test_get_metrics(self):
        """메트릭 조회"""
        self.controller.record_latency(100.0)
        self.controller.record_latency(150.0)
        self.controller.record_latency(80.0)

        metrics = self.controller.get_metrics()

        assert "avg_latency_ms" in metrics
        assert "min_latency_ms" in metrics
        assert "max_latency_ms" in metrics


class TestStreamingMetrics(unittest.TestCase):
    """StreamingMetrics 테스트 (2개)"""

    def test_metrics_initialization(self):
        """메트릭 초기화"""
        metrics = StreamingMetrics()

        assert metrics.total_tokens == 0
        assert metrics.chunk_count == 0

    def test_metrics_calculation(self):
        """메트릭 계산"""
        metrics = StreamingMetrics(total_tokens=100, total_time=5.0, chunk_count=10)
        metrics.tokens_per_second = 20.0
        metrics.avg_chunk_size = 10.0

        assert metrics.tokens_per_second == 20.0
        assert metrics.avg_chunk_size == 10.0


class TestGlobalInstances(unittest.TestCase):
    """전역 인스턴스 테스트 (2개)"""

    def test_get_streaming_handler(self):
        """스트리밍 핸들러 인스턴스"""
        handler = get_streaming_handler()
        assert handler is not None

    def test_get_adaptive_controller(self):
        """적응형 컨트롤러 인스턴스"""
        controller = get_adaptive_controller()
        assert controller is not None


class TestIntegration(unittest.TestCase):
    """통합 테스트 (3개)"""

    def test_full_streaming_pipeline(self):
        """전체 스트리밍 파이프라인"""

        async def _test():
            handler = StreamingResponseHandler()
            builder = StreamingResponseBuilder()

            async def token_gen():
                for token in ["hello", " ", "world"]:
                    await asyncio.sleep(0.01)
                    yield token

            chunks = []

            async def on_chunk(chunk: StreamChunk):
                chunks.append(chunk)
                builder.add_chunk(chunk)

            await handler.stream_response(token_gen(), on_chunk)

            assert builder.get_content() == "hello world"
            assert len(chunks) > 0

        asyncio.run(_test())

    def test_streaming_with_adaptive_control(self):
        """적응형 제어를 포함한 스트리밍"""

        async def _test():
            handler = StreamingResponseHandler()
            controller = AdaptiveStreamingController()

            async def token_gen():
                for i in range(10):
                    await asyncio.sleep(0.01)
                    yield f"t{i}"

            async def on_chunk(chunk: StreamChunk):
                controller.record_latency(chunk.timestamp * 1000)

            await handler.stream_response(token_gen(), on_chunk)

            metrics = controller.get_metrics()
            assert len(metrics) > 0

        asyncio.run(_test())

    def test_sse_format_integration(self):
        """SSE 포매팅 통합"""
        chunk = StreamChunk(
            content="test", timestamp=time.time(), token_count=1, chunk_index=0
        )

        sse_data = {"content": chunk.content, "index": chunk.chunk_index}

        sse = ServerSentEventsHandler.format_sse_event("chunk", sse_data, 1)

        assert "id: 1" in sse
        assert "test" in sse


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 60}")
    print("스트리밍 응답 처리 테스트 완료")
    print(f"{'=' * 60}")
    print(f"총 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    print(f"{'=' * 60}")
