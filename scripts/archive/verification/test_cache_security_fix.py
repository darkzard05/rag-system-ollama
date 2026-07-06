import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from src.core.rag_core import RAGSystem
from src.core.model_loader import ModelManager
from src.common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
from src.common.logging_config import setup_logging

async def test_cache_security_fix():
    setup_logging(log_level="INFO")
    
    print("\n" + "="*60)
    print("🧪 DiskCache Security Over-reaction Fix Verification")
    print("="*60)

    session_id = f"cache-test-{int(datetime.now().timestamp())}"
    rag = RAGSystem(session_id=session_id)
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)

    print("\n[STEP 1] Generating valid cache...")
    await rag.build_pipeline(test_pdf, file_name, embedder)
    
    from common.config import CACHE_DIR
    cache_dir = Path(CACHE_DIR) / "response_cache"
    cache_files = list(cache_dir.glob("*.cache"))
    if not cache_files:
        print("❌ No cache files found.")
        return

    target_cache = sorted(cache_files, key=os.path.getmtime)[-1]
    target_meta = Path(str(target_cache) + ".meta")

    if target_meta.exists():
        print(f"\n[STEP 2] Sabotaging cache: Deleting {target_meta.name}")
        os.remove(target_meta)
    else:
        print("❌ Metadata not found.")
        return

    print("\n[STEP 3] Retrying query with sabotaged cache...")
    try:
        result = await rag.aquery("What is this paper about?", model_name=DEFAULT_OLLAMA_MODEL)
        print(f"✅ Success! Response length: {len(result.get('response', ''))}")
        print("\n--- RESULTS ---")
        print("Check if the log above shows '[DiskCache] 유효하지 않은 캐시 항목 정리' (INFO)")
        print("instead of 'CRITICAL - 보안 위협 감지'.")
    except Exception as e:
        print(f"❌ Failed: {e}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_cache_security_fix())
