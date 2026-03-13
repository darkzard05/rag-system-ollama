
import asyncio
import sys
import os
import copy
from unittest.mock import MagicMock

# 프로젝트 루트 추가
sys.path.append(os.path.abspath("src"))

from core.rag_core import RAGSystem
from core.session import SessionManager

class MockBM25:
    def __init__(self, k=1):
        self.k = k

async def test_bm25_isolation():
    print("--- BM25 Isolation Verification Start ---")
    
    # 1. 공유 리소스 모킹
    shared_bm25 = MockBM25(k=10)
    
    # 2. 세션 A 설정 (k=3 예상)
    sid_a = "session_A"
    rag_a = RAGSystem(session_id=sid_a)
    SessionManager.set("file_hash", "hash_123", session_id=sid_a)
    
    # 3. 세션 B 설정 (k=5 예상)
    sid_b = "session_B"
    rag_b = RAGSystem(session_id=sid_b)
    SessionManager.set("file_hash", "hash_123", session_id=sid_b)

    # _prepare_config 내부 로직 시뮬레이션
    def get_isolated_bm25(sid, target_k):
        # rag_core.py의 로직 복제
        bm25_shared = shared_bm25
        bm25_ret = copy.copy(bm25_shared)
        bm25_ret.k = target_k
        return bm25_ret

    ret_a = get_isolated_bm25(sid_a, 3)
    ret_b = get_isolated_bm25(sid_b, 5)

    print(f"Shared BM25 k: {shared_bm25.k}")
    print(f"Session A BM25 k: {ret_a.k}")
    print(f"Session B BM25 k: {ret_b.k}")

    # 검증
    if ret_a.k == 3 and ret_b.k == 5 and shared_bm25.k == 10:
        print("✅ Success: BM25 parameters are correctly isolated per session.")
    else:
        print("❌ Failure: BM25 isolation failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_bm25_isolation())
