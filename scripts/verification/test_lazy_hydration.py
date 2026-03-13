import asyncio
import os
import time
from core.document_processor import load_pdf_docs
from core.rag_core import RAGSystem
from cache.coord_cache import coord_cache
from langchain_core.documents import Document

async def test_lazy_hydration_flow():
    pdf_path = "tests/data/2201.07520v1.pdf"
    file_name = os.path.basename(pdf_path)
    
    print(f"\n--- [Step 1] Fast Indexing (Lazy) ---")
    start_time = time.perf_counter()
    # 인덱싱 수행 (extract_words=False가 내부에서 작동함)
    docs = load_pdf_docs(pdf_path, file_name)
    elapsed = time.perf_counter() - start_time
    
    print(f"✅ Indexing completed in {elapsed:.2f}s")
    print(f"   (Compare to previous ~15s. Should be much faster now.)")
    
    # 첫 페이지 확인: has_coordinates는 True여야 하지만, 실제 word_coords는 없어야 함
    sample_doc = docs[2] # Page 3 (표가 있는 페이지)
    print(f"\n[Page {sample_doc.metadata['page']} Initial State]")
    print(f"- has_coordinates: {sample_doc.metadata.get('has_coordinates')}")
    print(f"- 'word_coords' in metadata: {'word_coords' in sample_doc.metadata}")
    
    # 캐시도 비어있어야 함 (첫 로딩이므로)
    file_hash = sample_doc.metadata['file_hash']
    cached_coords = coord_cache.get_coords(file_hash, sample_doc.metadata['page'])
    print(f"- Coords in Cache: {cached_coords is not None}")

    print(f"\n--- [Step 2] On-demand Hydration ---")
    rag = RAGSystem(session_id="test_lazy")
    
    # 검색 결과로 3페이지만 선택되었다고 가정
    retrieved_docs = [sample_doc]
    
    start_hydrate = time.perf_counter()
    # _hydrate_docs 호출 (이때 PDF를 열어 좌표를 추출해야 함)
    rag._hydrate_docs(retrieved_docs)
    hydrate_elapsed = time.perf_counter() - start_hydrate
    
    print(f"✅ Hydration completed in {hydrate_elapsed:.4f}s")
    print(f"[Page {sample_doc.metadata['page']} After Hydration]")
    print(f"- 'word_coords' in metadata: {'word_coords' in sample_doc.metadata}")
    if 'word_coords' in sample_doc.metadata:
        print(f"- Extracted Coords Count: {len(sample_doc.metadata['word_coords'])}")
        print(f"- Sample Coord: {sample_doc.metadata['word_coords'][0]}")
    
    # 캐시 확인
    cached_coords = coord_cache.get_coords(file_hash, sample_doc.metadata['page'])
    print(f"- Coords in Cache Now: {cached_coords is not None}")

if __name__ == "__main__":
    # 캐시 초기화 (정확한 테스트를 위해)
    # 실제 환경에서는 캐시가 있으면 더 빠르겠지만, 로직 검증을 위해 비우고 시작 권장
    asyncio.run(test_lazy_hydration_flow())
