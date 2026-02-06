import ollama
import time
import re

def test_bge_reranker():
    model_name = 'qllama/bge-reranker-v2-m3:f16'
    print(f"🚀 [BGE-Reranker f16] 테스트 시작: {model_name}")
    
    test_cases = [
        {
            "query": "What is the capital of France?",
            "doc": "Paris is the capital and most populous city of France, situated on the Seine River."
        },
        {
            "query": "What is the capital of France?",
            "doc": "The history of chocolate began in Mesoamerica, where fermented beverages were made from cocoa."
        }
    ]

    try:
        for i, case in enumerate(test_cases):
            print(f"\n📝 Case {i+1} 추론 중...")
            start_time = time.time()
            prompt = "Query: " + case['query'] + "\nDocument: " + case['doc'] + "\nRelevance Score (0.0-1.0):"
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={'temperature': 0, 'num_predict': 20}
            )
            
            content = response['response'].strip()
            elapsed = time.time() - start_time
            
            scores = re.findall(r"0\.\d+|1\.0|\d\.\d+", content)
            final_score = scores[0] if scores else "N/A"
            
            print("✅ 완료 (" + str(round(elapsed, 2)) + "s)")
            print("📥 응답 원문: '" + content + "'")
            print("🎯 추출된 점수: " + str(final_score))

    except Exception as e:
        print("❌ 실패: " + str(e))

if __name__ == "__main__":
    test_bge_reranker()