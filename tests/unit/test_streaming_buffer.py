import time

from api.streaming_handler import TokenStreamBuffer


def test_buffer_size_flush():
    """TTFT 첫 토큰 즉시 플러시 후, buffer_size개 모이면 일괄 플러시."""
    buffer = TokenStreamBuffer(buffer_size=3, timeout_ms=1000.0)

    # 첫 번째 토큰은 TTFT 최적화로 즉시 플러시
    assert buffer.add_token("1") == "1"
    # 두 번째부터 버퍼링
    assert buffer.add_token("2") is None
    assert buffer.add_token("3") is None
    # 네 번째 → 버퍼(2,3,4) 플러시
    result = buffer.add_token("4")
    assert result == "234"
    assert len(buffer.buffer) == 0


def test_timeout_flush():
    """timeout_ms 경과 후 토큰 추가 시 미리 쌓인 버퍼가 플러시됨."""
    buffer = TokenStreamBuffer(buffer_size=10, timeout_ms=100.0)

    assert buffer.add_token("1") == "1"
    assert buffer.add_token("2") is None
    time.sleep(0.15)

    result = buffer.add_token("3")
    assert result == "23"
    assert len(buffer.buffer) == 0


def test_reset():
    """reset 호출 시 버퍼와 카운트가 초기화됨."""
    buffer = TokenStreamBuffer(buffer_size=3)
    buffer.add_token("a")
    buffer.reset()
    assert len(buffer.buffer) == 0
    assert buffer.token_count == 0
