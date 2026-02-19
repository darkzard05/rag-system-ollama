import ollama
import time
import re

def test_llm_as_reranker():
    model_name = 'qwen3:4b-instruct-2507-q4_K_M'
    print("🚀 [LLM-Reranker] 테스트 시작: " + model_name)
    
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
            print("\n📝 Case " + str(i+1) + " 채점 중...")
            start_time = time.time()
            
            prompt = "Assess relevance between query and document.\n"
            prompt += "Query: " + case['query'] + "\n"
            prompt += "Document: " + case['doc'] + "\n\n"
            prompt += "Output ONLY a single number between 0.0 and 1.0 representing relevance."

            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0}
            )
            
            content = response['message']['content'].strip()
            elapsed = time.time() - start_time
            
            scores = re.findall(r"0\.\d+|1\.0|\d\.\d+", content)
            final_score = scores[0] if scores else "N/A"
            
            print("✅ 완료 (" + str(round(elapsed, 2)) + "s)")
            print("📥 LLM 답변: '" + content + "'")
            print("🎯 최종 추출 점수: " + str(final_score))

    except Exception as e:
        print("❌ 실패: " + str(e))

if __name__ == "__main__":
    test_llm_as_reranker()