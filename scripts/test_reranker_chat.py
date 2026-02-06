from ollama import chat
import time

def test_reranker_chat_api():
    model_name = 'medaibase/qwen3-vl-reranker:2b'
    print(f"🚀 [Chat API] 테스트 시작 (Lowercase): {model_name}")
    
    query = "What is the capital of France?"
    doc = "Paris is the capital and most populous city of France."
    
    prompt = "Evaluate the relevance of the document to the query.\n"
    prompt += f"Query: {query}\n"
    prompt += f"Document: {doc}\n\n"
    prompt += "Output ONLY a single relevance score between 0.0 and 1.0."

    start_time = time.time()
    try:
        print("⏳ 모델 로딩 및 Chat 응답 대기 중...")
        response = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 성공! (소요 시간: {elapsed:.2f}s)")
        # 속성 접근 및 딕셔너리 접근 모두 지원
        try:
            content = response.message.content
        except AttributeError:
            content = response['message']['content']
            
        print(f"📥 응답 내용: {content.strip()}")
        
    except Exception as e:
        print(f"❌ 실패: {e}")

if __name__ == "__main__":
    test_reranker_chat_api()