import ollama
import time

def test_specific_reranker():
    model_name = "qllama/bge-reranker-v2-m3"
    print(f"🚀 테스트 시작: {model_name}")
    
    query = "What is the capital of France?"
    doc = "Paris is the capital and most populous city of France."
    
    # 1. 전용 API 테스트
    print("⏳ [Step 1] /api/rank API 테스트 시도 중...")
    try:
        import requests
        res = requests.post("http://127.0.0.1:11434/api/rank", json={
            "model": model_name,
            "query": query,
            "documents": [doc]
        }, timeout=30)
        
        if res.status_code == 200:
            print(f"✅ /api/rank 성공! 점수: {res.json()['results'][0]['score']}")
            return
        else:
            print(f"⚠️ /api/rank 미지원 (Status: {res.status_code})")
    except Exception as e:
        print(f"⚠️ /api/rank 오류: {e}")

    # 2. 일반 Generate 테스트 (폴백)
    print("\n⏳ [Step 2] 일반 generate 추론 시도 중...")
    import ollama
    prompt = f"Query: {query}\n\nDocument: {doc}\n\nRelevance Score (0.0-1.0):"
    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"num_predict": 10})
        print(f"✅ 성공! 응답: {response['response'].strip()}")
    except Exception as e:
        print(f"❌ 실패: {e}")

if __name__ == "__main__":
    test_specific_reranker()