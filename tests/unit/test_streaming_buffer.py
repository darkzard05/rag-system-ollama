import time
import unittest
import sys
from pathlib import Path

# --- 경로 설정 최적화 (절대 경로 기반) ---
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.streaming_handler import TokenStreamBuffer


class TestTokenStreamBuffer(unittest.TestCase):
    def test_buffer_size_flush(self):
        # buffer_size=3, timeout=1000ms
        buffer = TokenStreamBuffer(buffer_size=3, timeout_ms=1000.0)

        # 첫 토큰은 TTFT 최적화로 인해 즉시 플러시됨
        assert buffer.add_token("a") == "a"
        # 두 번째, 세 번째는 버퍼링됨
        assert buffer.add_token("b") is None
        assert buffer.add_token("c") is None
        # 네 번째 추가 시 버퍼가 꽉 차서(b, c, d) 플러시됨
        result = buffer.add_token("d")
        assert result == "bcd"
        assert len(buffer.buffer) == 0

    def test_timeout_flush(self):
        # buffer_size=10, timeout=100ms
        buffer = TokenStreamBuffer(buffer_size=10, timeout_ms=100.0)

        # 첫 토큰은 즉시 플러시
        assert buffer.add_token("a") == "a"
        # 두 번째 토큰 추가 후 대기
        assert buffer.add_token("b") is None
        time.sleep(0.15)  # Wait for > 100ms

        # 세 번째 토큰 추가 시 타임아웃 플러시 발생 (b + c)
        result = buffer.add_token("c")
        assert result == "bc"
        assert len(buffer.buffer) == 0

    def test_reset(self):
        buffer = TokenStreamBuffer(buffer_size=3)
        buffer.add_token("a")
        buffer.reset()
        assert len(buffer.buffer) == 0


if __name__ == "__main__":
    unittest.main()