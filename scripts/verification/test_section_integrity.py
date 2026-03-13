import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.document_processor import load_pdf_docs
from core.chunking import split_documents
from core.graph_builder import _merge_adjacent_chunks
from core.model_loader import ModelManager
from common.config import DEFAULT_EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_section_integrity():
    """실제 PDF를 이용해 섹션 추출 및 병합 무결성을 검증합니다."""
    
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"테스트용 PDF가 없습니다: {pdf_path}")
        return

    logger.info(f"=== 실전 섹션 무결성 테스트 시작: {os.path.basename(pdf_path)} ===")

    # 1. 문서 로드 및 청킹 (의미론적 청킹 활성화)
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    docs = load_pdf_docs(pdf_path, os.path.basename(pdf_path))
    
    # split_documents를 통해 상세 섹션 정보 수집
    split_docs, _ = await split_documents(docs, embedder)
    
    logger.info(f"\n[1. 섹션 추출 결과 확인]")
    unique_sections = []
    for d in split_docs:
        sec = d.metadata.get("current_section", "N/A")
        if sec not in unique_sections:
            unique_sections.append(sec)
            logger.info(f"- 발견된 섹션: {sec}")

    # 2. 섹션 경계 병합 테스트
    logger.info(f"\n[2. 섹션 경계 병합 보호 테스트]")
    
    # 인위적으로 섹션 경계 상황 연출 (페이지와 오프셋은 인접하지만 섹션이 다름)
    doc_intro_end = split_docs[0]
    doc_intro_end.metadata["current_section"] = "INTRODUCTION"
    doc_intro_end.metadata["end_index"] = 1000
    
    from langchain_core.documents import Document
    import copy
    
    doc_method_start = Document(
        page_content="새로운 섹션의 시작 문장입니다.",
        metadata=copy.copy(doc_intro_end.metadata)
    )
    doc_method_start.metadata["current_section"] = "METHOD" # 섹션 변경
    doc_method_start.metadata["start_index"] = 1001 # 오프셋은 인접 (1자 차이)
    
    test_batch = [doc_intro_end, doc_method_start]
    merged = _merge_adjacent_chunks(test_batch, max_tokens=2000)
    
    logger.info(f"결과: {len(merged)}개의 청크로 유지됨")
    if len(merged) == 2:
        logger.info("✅ 성공: 섹션이 다르면 인접해도 병합되지 않습니다. (주제 일관성 유지)")
    else:
        logger.error("❌ 실패: 서로 다른 섹션임에도 인접성 때문에 강제로 병합되었습니다.")

    # 3. 실제 데이터 기반 샘플링 (첫 5개 청크 병합 시도)
    logger.info(f"\n[3. 실제 데이터 병합 샘플링]")
    actual_merged = _merge_adjacent_chunks(split_docs[:10], max_tokens=1500)
    logger.info(f"원본 {min(10, len(split_docs))}개 청크 -> 병합 후 {len(actual_merged)}개 섹션으로 압축됨")
    
    for i, m in enumerate(actual_merged):
        logger.info(f"병합 섹션 {i+1}: [{m.metadata.get('current_section')}] (길이: {len(m.page_content)})")

if __name__ == "__main__":
    asyncio.run(verify_section_integrity())
