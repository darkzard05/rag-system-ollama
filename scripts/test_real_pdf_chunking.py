
import asyncio
import sys
import os
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.document_processor import load_pdf_docs
from core.semantic_chunker import EmbeddingBasedSemanticChunker

class RobustFakeEmbeddings:
    def __init__(self, size=1536):
        self.size = size
        self.model_name = "unique_test_model_v1" # 캐시 충돌 방지
    def embed_documents(self, texts):
        np.random.seed(42)
        return [np.random.rand(self.size).tolist() for _ in texts]
    def embed_query(self, text):
        return np.random.rand(self.size).tolist()

async def test_real_pdf_chunking():
    pdf_path = "tests/data/2201.07520v1.pdf"
    file_name = os.path.basename(pdf_path)
    
    print(f"\n[1/3] PDF 파싱 중: {file_name}")
    docs = load_pdf_docs(pdf_path, file_name)
    print(f"파싱 완료: {len(docs)}페이지 확보")

    # 캐시 충돌을 방지하기 위해 유니크한 모델명 부여
    embedder = RobustFakeEmbeddings(size=1536)
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        max_chunk_size=1000,
        similarity_threshold=0.6,
        chunk_overlap=1
    )

    print(f"\n[2/3] Header-Aware Semantic Chunking 실행 중...")
    all_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 디버깅: sentences 추출 단계 확인
    chunks = await chunker.split_text(all_text)
    print(f"청킹 완료: 총 {len(chunks)}개 청크 생성")

    print("\n[3/3] 섹션별 청킹 결과 분석 (Top 15):")
    print("-" * 80)
    for i, chunk in enumerate(chunks[:20]):
        section = chunk.get("current_section", "Intro/Root")
        content = chunk['text'].strip()
        
        has_header = False
        lines = content.split('\n')
        first_header = ""
        for line in lines[:3]:
            if line.strip().startswith('#'):
                has_header = True
                first_header = line.strip()
                break
        
        marker = "✅ [Header-Start]" if has_header else "📄 [Content]"
        print(f"Chunk {i+1:2d} | {marker} | Section: {section[:40]:<40} | Size: {len(content):5d}자")
        if has_header:
            print(f"   └─ Detected Header: {first_header}")

if __name__ == "__main__":
    asyncio.run(test_real_pdf_chunking())
