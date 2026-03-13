import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.documents import Document
from core.graph_builder import _merge_adjacent_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_integrity_fix():
    """
    개선된 오프셋 기반 병합 로직의 무결성을 검증합니다.
    """
    logger.info("=== 문맥 무결성 개선 로직 검증 시작 ===")
    
    # 1. [실패 시나리오] 중간 내용이 유실된 청크들 (병합되면 안 됨)
    # A(0~100) -> B(101~500, 삭제됨) -> C(501~600)
    doc_a = Document(
        page_content="이것은 서론의 마지막 문장입니다.",
        metadata={
            "source": "test.pdf", "page": 1, "chunk_index": 0,
            "start_index": 0, "end_index": 100
        }
    )
    doc_c = Document(
        page_content="이것은 결론의 시작 문장입니다.",
        metadata={
            "source": "test.pdf", "page": 1, "chunk_index": 1, # 인덱스는 연속적임
            "start_index": 501, "end_index": 600 # 오프셋은 401자의 간격이 있음
        }
    )
    
    logger.info("시나리오 1: 중간 텍스트 간극(Gap)이 큰 청크들의 병합 여부 확인")
    res1 = _merge_adjacent_chunks([doc_a, doc_c], max_tokens=1000)
    
    if len(res1) == 2:
        logger.info("✅ 성공: 텍스트 간극이 커서 병합되지 않았습니다. (무결성 유지)")
    else:
        logger.error("❌ 실패: 여전히 잘못된 병합이 발생하고 있습니다.")

    # 2. [성공 시나리오] 실제 인접한 청크들 (병합되어야 함)
    # A(0~100) -> B(101~200)
    doc_a2 = Document(
        page_content="첫 번째 문장.",
        metadata={
            "source": "test.pdf", "page": 1, "chunk_index": 0,
            "start_index": 0, "end_index": 100
        }
    )
    doc_b2 = Document(
        page_content="두 번째 문장.",
        metadata={
            "source": "test.pdf", "page": 1, "chunk_index": 1,
            "start_index": 101, "end_index": 200
        }
    )
    
    logger.info("시나리오 2: 실제로 인접한 청크들의 병합 여부 확인")
    res2 = _merge_adjacent_chunks([doc_a2, doc_b2], max_tokens=1000)
    
    if len(res2) == 1:
        logger.info("✅ 성공: 인접한 청크들이 올바르게 병합되었습니다.")
    else:
        logger.error("❌ 실패: 인접한 청크들이 병합되지 않았습니다. (기능 퇴화)")

if __name__ == "__main__":
    asyncio.run(verify_integrity_fix())
