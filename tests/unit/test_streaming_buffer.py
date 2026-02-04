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