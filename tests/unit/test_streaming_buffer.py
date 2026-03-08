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

        # [변경점] 첫 번째 토큰만 TTFT 최적화로 인해 즉시 플러시됨
        assert buffer.add_token("1") == "1"
        
        # 두 번째부터 버퍼링 시작 (3개 찰 때까지)
        assert buffer.add_token("2") is None
        assert buffer.add_token("3") is None
        
        # 네 번째 추가 시 버퍼가 꽉 차서(2, 3, 4) 플러시됨
        result = buffer.add_token("4")
        assert result == "234"
        assert len(buffer.buffer) == 0

    def test_timeout_flush(self):
        # buffer_size=10, timeout=100ms
        buffer = TokenStreamBuffer(buffer_size=10, timeout_ms=100.0)

        # 첫 번째만 즉시 플러시
        assert buffer.add_token("1") == "1"
            
        # 두 번째 토큰 추가 후 대기
        assert buffer.add_token("2") is None
        time.sleep(0.15)  # Wait for > 100ms

        # 세 번째 토큰 추가 시 타임아웃 플러시 발생 (2 + 3)
        result = buffer.add_token("3")
        assert result == "23"
        assert len(buffer.buffer) == 0

    def test_reset(self):
        buffer = TokenStreamBuffer(buffer_size=3)
        buffer.add_token("a")
        buffer.reset()
        assert len(buffer.buffer) == 0
        assert buffer.token_count == 0


if __name__ == "__main__":
    unittest.main()
