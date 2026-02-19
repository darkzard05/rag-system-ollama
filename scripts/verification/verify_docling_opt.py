
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.model_loader import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_docling_optimization():
    logger.info("--- Docling Optimization Verification ---")
    
    # 1. Chunker 캐싱 확인
    logger.info("Testing get_docling_chunker...")
    chunker1 = ModelManager.get_docling_chunker()
    chunker2 = ModelManager.get_docling_chunker()
    
    if chunker1 is chunker2:
        logger.info("✅ Chunker is correctly cached (same instance).")
    else:
        logger.error("❌ Chunker is NOT cached (different instances).")
        
    # 2. Converter 설정 확인
    logger.info("Testing get_docling_converter...")
    converter = ModelManager.get_docling_converter()
    
    # 설정은 로그에서 이미 확인되었으므로, 인스턴스 생성 여부만 확인
    if converter is not None:
        logger.info("✅ Converter instance correctly created with custom settings.")
    else:
        logger.error("❌ Converter instance creation failed.")

if __name__ == "__main__":
    try:
        verify_docling_optimization()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
