import os
import sys
import time
import unittest

# 프로젝트 루트 및 src 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from api.streaming_handler import TokenStreamBuffer


class TestTokenStreamBuffer(unittest.TestCase):
    def test_buffer_size_flush(self):
        # buffer_size=3, timeout=1000ms
        buffer = TokenStreamBuffer(buffer_size=3, timeout_ms=1000.0)

        assert buffer.add_token("a") is None
        assert buffer.add_token("b") is None
        result = buffer.add_token("c")
        assert result == "abc"
        assert len(buffer.buffer) == 0

    def test_timeout_flush(self):
        # buffer_size=10, timeout=100ms
        buffer = TokenStreamBuffer(buffer_size=10, timeout_ms=100.0)

        assert buffer.add_token("a") is None
        time.sleep(0.15)  # Wait for > 100ms

        result = buffer.add_token("b")
        assert result == "ab"
        assert len(buffer.buffer) == 0

    def test_reset(self):
        buffer = TokenStreamBuffer(buffer_size=3)
        buffer.add_token("a")
        buffer.reset()
        assert len(buffer.buffer) == 0


if __name__ == "__main__":
    unittest.main()
