
import time
import torch
import numpy as np
from sentence_transformers import CrossEncoder
import os

def benchmark_reranker():
    model_name = "BAAI/bge-reranker-v2-m3"
    
    print(f"=== 리랭커(Reranker) 하드웨어 벤치마크 ===")
    print(f"모델: {model_name}")
    
    # 1. 모델 로드 시간 측정
    print("\n1. 모델 로딩 중...")
    start_load = time.time()
    try:
        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {device}")
        
        model = CrossEncoder(model_name, device=device)
        load_duration = time.time() - start_load
        print(f"로딩 완료! 소요 시간: {load_duration:.2f}초")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("참고: 인터넷 연결이 필요하거나 메모리가 부족할 수 있습니다.")
        return

    # 2. 추론 속도 측정 (다양한 배치 사이즈)
    query = "What are the key benefits of using RAG with local LLMs?"
    
    # 가상의 검색 결과 (12개 문서)
    documents = [
        "Local LLMs provide better data privacy as information doesn't leave your network.",
        "RAG helps reduce hallucinations by providing external context to the model.",
        "System performance depends heavily on the available GPU VRAM.",
        "Ollama is a popular tool for running large language models locally.",
        "Vector databases like FAISS are essential for efficient similarity search.",
        "Semantic chunking improves retrieval quality by keeping related sentences together.",
        "Cross-encoders are more accurate but slower than bi-encoders for ranking.",
        "Hybrid search combines keyword-based and semantic search strategies.",
        "The weather today is sunny with a chance of rain in the evening.", # Noise
        "Cooking pasta requires boiling water and adding salt for flavor.", # Noise
        "Recursive character text splitting is a common baseline chunking method.",
        "GPU acceleration significantly speeds up both embedding and inference."
    ]

    print(f"\n2. 추론 성능 테스트 (대상 문서: {len(documents)}개)")
    
    # Warm-up
    model.predict([query, documents[0]])
    
    # 실제 측정
    start_inference = time.time()
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    inference_duration = time.time() - start_inference
    
    print(f"총 추론 시간: {inference_duration*1000:.2f}ms")
    print(f"문서당 평균 시간: {(inference_duration/len(documents))*1000:.2f}ms")

    # 3. 순위 변화 분석
    ranked_indices = np.argsort(scores)[::-1]
    
    print("\n3. 리랭킹 결과 (상위 5개):")
    for i, idx in enumerate(ranked_indices[:5], 1):
        print(f"[{i}위] Score: {scores[idx]:.4f} | {documents[idx]}")

    # 4. 결론 도출
    print("\n=== 종합 평가 ===")
    if inference_duration < 0.5:
        print("상태: [매우 우수] 지연 시간이 매우 낮습니다. 상시 활성화를 강력 추천합니다.")
    elif inference_duration < 1.0:
        print("상태: [우수] 실시간 대화에 지장이 없는 수준입니다. 활성화를 추천합니다.")
    elif inference_duration < 2.0:
        print("상태: [보통] 약간의 체감 지연이 있으나 정확도가 중요하다면 사용하세요.")
    else:
        print("상태: [느림] 로컬 하드웨어 부하가 큽니다. 꼭 필요한 경우에만 사용하거나 대상을 줄이세요.")

if __name__ == "__main__":
    benchmark_reranker()
