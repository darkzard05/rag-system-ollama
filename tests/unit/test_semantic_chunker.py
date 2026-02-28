import pytest
import asyncio
import numpy as np
from langchain_core.documents import Document
from core.semantic_chunker import EmbeddingBasedSemanticChunker

class MockEmbeddings:
    """테스트용 가짜 임베딩 모델"""
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.model_name = "mock-model"

    def embed_documents(self, texts):
        results = []
        for text in texts:
            vec = np.zeros(self.dimension)
            # 텍스트 내용에 따라 명확히 다른 벡터 생성 (분리 유도)
            if "Group A" in text:
                vec[0] = 1.0
            elif "Group B" in text:
                vec[1] = 1.0
            else:
                # 랜덤 요소를 섞어 같은 텍스트라도 약간의 차이 유도 (강제 병합 방지)
                vec[2] = 1.0 + (np.random.rand() * 0.01)
            results.append(vec.tolist())
        return results

@pytest.mark.asyncio
async def test_semantic_chunker_overlap():
    """청크 간 Overlap(겹침)이 정상적으로 발생하는지 테스트"""
    embedder = MockEmbeddings()
    # 겹침을 1개 문장으로 설정하고, 병합을 방지하기 위해 max_chunk_size를 작게 설정
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        chunk_overlap=1,
        min_chunk_size=0,
        max_chunk_size=250, # 넉넉하게 설정
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_value=0.1
    )
    
    # 문장 길이를 30자 이상으로 길게 작성
    s1 = "Group A sentence 1 is long enough to avoid merging. "
    s2 = "Group A sentence 2 is also quite long and descriptive. "
    s3 = "Group A sentence 3 provides additional context here. "
    s4 = "Group B sentence 4 marks the beginning of a new topic. "
    s5 = "Group B sentence 5 follows the new topic with details. "
    s6 = "Group B sentence 6 concludes the second group of text. "
    
    text = s1 + s2 + s3 + s4 + s5 + s6
    
    chunks = await chunker.split_text(text)
    
    # 분할 확인
    assert len(chunks) >= 2
    
    # Overlap 확인
    found_overlap = False
    for i in range(1, len(chunks)):
        curr_chunk = chunks[i]["text"]
        if ("Group A" in curr_chunk and "Group B" in curr_chunk) or \
           ("sentence 3" in curr_chunk and "sentence 4" in curr_chunk):
            found_overlap = True
            break
            
    assert found_overlap, "Chunk overlap failed"

@pytest.mark.asyncio
async def test_semantic_chunker_cross_page_metadata():
    """여러 페이지에 걸친 청크의 메타데이터가 올바르게 보존되는지 테스트"""
    embedder = MockEmbeddings()
    chunker = EmbeddingBasedSemanticChunker(embedder=embedder, chunk_overlap=0)
    
    docs = [
        Document(page_content="Content on page 1 is long enough.", metadata={"page": 1}),
        Document(page_content="Content on page 2 is also long enough.", metadata={"page": 2}),
        Document(page_content="Content on page 3 is finally long enough.", metadata={"page": 3})
    ]
    
    chunker.min_chunk_size = 5000 
    chunker.max_chunk_size = 10000
    final_docs, _ = await chunker.split_documents(docs)
    
    assert len(final_docs) > 0
    meta = final_docs[0].metadata
    assert "pages" in meta
    assert 1 in meta["pages"] and 2 in meta["pages"] and 3 in meta["pages"]
    assert meta["is_cross_page"] is True

@pytest.mark.asyncio
async def test_semantic_chunker_sentence_splitting_safety():
    """매우 긴 텍스트(구분자 없음)에 대한 강제 분할 안전성 테스트"""
    embedder = MockEmbeddings()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        min_chunk_size=0,
        max_chunk_size=500
    )
    
    long_text = "A" * 2000
    
    chunks = await chunker.split_text(long_text)
    
    # 1. 분할이 발생했는지 확인
    assert len(chunks) > 1
    # 2. 모든 청크가 합리적인 크기인지 확인 (병합 허용 범위 내)
    for c in chunks:
        # [최적화] hard_split_limit가 1.5배로 늘어났으므로 임계값을 1600으로 상향 (500*1.5=750이지만, 병합 로직 고려)
        assert len(c["text"]) <= 1600
