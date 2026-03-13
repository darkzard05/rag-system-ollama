
import asyncio
import sys
import os
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.abspath("src"))

from core.resource_pool import get_resource_pool

# 로그 설정
logging.basicConfig(level=logging.INFO)

async def test_resource_pool_lru():
    print("--- ResourcePool LRU Test Start ---")
    
    # max_size=2로 설정하여 테스트
    pool = get_resource_pool()
    pool.max_size = 2
    
    # 1. 리소스 3개 등록 (마지막에 등록된 2개만 남아야 함)
    print("Registering 3 resources (Limit=2)...")
    await pool.register("hash1", "vs1", "bm1")
    await pool.register("hash2", "vs2", "bm2")
    await pool.register("hash3", "vs3", "bm3")
    
    # 2. 결과 확인 (hash1은 제거되어야 함)
    vs1, _ = await pool.get("hash1")
    vs2, _ = await pool.get("hash2")
    vs3, _ = await pool.get("hash3")
    
    print(f"Resource 1 exists: {vs1 is not None} (Expected: False)")
    print(f"Resource 2 exists: {vs2 is not None} (Expected: True)")
    print(f"Resource 3 exists: {vs3 is not None} (Expected: True)")
    
    if vs1 is None and vs2 == "vs2" and vs3 == "vs3":
        print("✅ Success: LRU logic is working correctly.")
    else:
        print("❌ Failure: Resource pool logic failed.")
        sys.exit(1)
    
    # 3. 명시적 제거 테스트
    print("Unregistering Resource 2...")
    await pool.unregister("hash2")
    vs2_after, _ = await pool.get("hash2")
    print(f"Resource 2 exists after unregister: {vs2_after is not None} (Expected: False)")
    
    if vs2_after is None:
        print("✅ Success: Unregister logic is working correctly.")
    else:
        print("❌ Failure: Unregister logic failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_resource_pool_lru())
