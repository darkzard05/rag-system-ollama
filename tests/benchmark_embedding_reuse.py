import time
import sys
from pathlib import Path
from typing import List

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from langchain_core.documents import Document
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from langchain_community.vectorstores import FAISS


# 호출 횟수를 기록하기 위한 가짜 임베딩 모델
class MockEmbedder:
    def __init__(self):
        self.call_count = 0
        self.total_embedded_texts = 0
        self.dim = 384  # MiniLM 차원

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self.total_embedded_texts += len(texts)
        # 실제 LLM/GPU 연산 부하를 시뮬레이션 (최소 0.1초 + 토큰당 지연)
        time.sleep(0.1 + len(texts) * 0.005)
        return [[0.1] * self.dim for _ in texts]


def run_benchmark():
    print("--- 임베딩 재사용 최적화 성능 테스트 (대규모 데이터) ---")

    # 테스트 데이터 대폭 확대 (1,000문장)
    test_sentences = [
        f"이것은 실제와 유사한 성능 측정을 위해 생성된 {i}번째 문장입니다. 문장이 길어질수록 최적화 효과가 극대화됩니다."
        for i in range(1000)
    ]
    test_text = " ".join(test_sentences)
    docs = [Document(page_content=test_text, metadata={"source": "large_doc.pdf"})]

    # 1. 이전 방식 시뮬레이션 (Double Embedding)
    print("\n[1] 최적화 이전 방식 (Double Embedding) 시뮬레이션 중...")
    embedder_legacy = MockEmbedder()
    chunker_legacy = EmbeddingBasedSemanticChunker(
        embedder=embedder_legacy, min_chunk_size=100, max_chunk_size=500
    )

    start_time = time.time()
    # Step A: 청킹 (임베딩 1회 발생)
    # 구버전은 Document만 반환하므로 내부 로직을 직접 실행하여 시뮬레이션
    _ = chunker_legacy.split_documents(docs)

    # Step B: 벡터 저장소 생성 (임베딩 1회 추가 발생)
    texts_to_embed = [f"chunk {i}" for i in range(20)]  # 약 20개 청크 가정
    _ = embedder_legacy.embed_documents(texts_to_embed)

    legacy_time = time.time() - start_time
    legacy_calls = embedder_legacy.call_count

    # 2. 현재 방식 (Optimized - Vector Reuse)
    print("[2] 최적화된 현재 방식 (Vector Reuse) 측정 중...")
    embedder_opt = MockEmbedder()
    chunker_opt = EmbeddingBasedSemanticChunker(
        embedder=embedder_opt, min_chunk_size=100, max_chunk_size=500
    )

    start_time = time.time()
    # Step A: 청킹 및 벡터 획득 (임베딩 1회 발생)
    split_docs, vectors = chunker_opt.split_documents(docs)

    # Step B: 벡터 재사용하여 FAISS 생성 (임베딩 발생 0회)
    if vectors:
        text_embeddings = zip([d.page_content for d in split_docs], vectors)
        _ = FAISS.from_embeddings(
            text_embeddings, embedder_opt, metadatas=[d.metadata for d in split_docs]
        )

    opt_time = time.time() - start_time
    opt_calls = embedder_opt.call_count

    # 결과 출력
    print("\n" + "=" * 50)
    print(f"{'.':<20} | {'이전 방식':<12} | {'현재 방식':<12}")
    print("-" * 50)
    print(f"{'.':<20} | {legacy_calls:<14} | {opt_calls:<14}")
    print(f"{'.':<20} | {legacy_time:<14.4f} | {opt_time:<14.4f}")
    print("-" * 50)

    improvement = (legacy_time - opt_time) / legacy_time * 100
    print(f"속도 개선율: {improvement:.2f}%")
    print(
        f"호출 감소량: {legacy_calls - opt_calls}회 ({(legacy_calls - opt_calls) / legacy_calls * 100:.0f}% 감소)"
    )
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
