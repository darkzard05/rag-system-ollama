
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from langchain_community.embeddings import FakeEmbeddings

async def run_comparison():
    # 가짜 임베딩 모델 (속도를 위해)
    embedder = FakeEmbeddings(size=1536)
    
    # 헤더 인식이 포함된 현재 버전의 청커
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        max_chunk_size=500,
        similarity_threshold=0.5
    )

    # 테스트 데이터 (헤더가 포함된 논문 스타일 텍스트)
    sample_text = """
# 1. Introduction
This is the first sentence of the introduction. Deep learning has revolutionized many fields. 
The second sentence explains how it works.

# 2. Methodology
We propose a new method for RAG systems. This involves several steps.
First, we analyze the structure. Second, we apply semantic chunking.

## 2.1 Preprocessing
Data cleaning is the first step. We remove noise and outliers.
The results show improved accuracy.
    """.strip()

    print("\n=== [Header-Aware Chunking Result] ===")
    chunks = await chunker.split_text(sample_text)
    
    for i, chunk in enumerate(chunks):
        section = chunk.get("current_section", "Unknown")
        print(f"\n[Chunk {i+1}] (Section: {section})")
        print(f"Content: {chunk['text'][:100]}...")
        
        # 헤더가 청크의 중간이 아닌 시작점에 있는지 확인
        lines = chunk['text'].strip().split('\n')
        for j, line in enumerate(lines):
            if line.startswith('#'):
                if j == 0:
                    print(f"✅ Success: Header '{line}' found at the start of chunk.")
                else:
                    print(f"❌ Warning: Header '{line}' found at line {j+1} of chunk.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
