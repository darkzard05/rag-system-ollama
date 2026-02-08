
import asyncio
import logging
import time
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL, INTENT_PARAMETERS
from core.session import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def evaluate_params():
    print("=== Intent Parameter Evaluation ===")
    
    rag = RAGSystem(session_id="eval-session")
    embedder = load_embedding_model()
    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    await rag.load_document(test_pdf, "eval_doc.pdf", embedder)

    # 테스트 케이스: (의도, 질문)
    test_cases = [
        ("FACTOID", "What is the name of the model introduced in this paper?"),
        ("RESEARCH", "Compare the performance of CM3 with other models like DALL-E."),
        ("SUMMARY", "Provide a comprehensive summary of the entire paper including methodology and results.")
    ]

    # K값 변조 리스트
    k_variations = [0.5, 1.0, 1.5] # 기본값의 50%, 100%, 150%
    
    all_results = []

    for intent, query in test_cases:
        base_params = INTENT_PARAMETERS.get(intent, INTENT_PARAMETERS["DEFAULT"])
        
        for multiplier in k_variations:
            test_k = int(base_params["retrieval_k"] * multiplier)
            test_top_k = int(base_params["rerank_top_k"] * multiplier)
            
            print(f"\n[Test] Intent: {intent} | Multiplier: {multiplier} (K: {test_k}, Top-K: {test_top_k})")
            
            # 파라미터 강제 설정 (세션 및 전역 설정을 흉내내어 주입)
            # 실제로는 rag.aquery 내부의 config 구성을 조작해야 하지만, 
            # 테스트를 위해 임시로 INTENT_PARAMETERS를 직접 수정 (테스트 스크립트 한정)
            original_k = base_params["retrieval_k"]
            original_top_k = base_params["rerank_top_k"]
            
            base_params["retrieval_k"] = test_k
            base_params["rerank_top_k"] = test_top_k
            
            try:
                start_time = time.time()
                # 의도 분석 노드를 우회하기 위해 input에 힌트를 주거나 
                # RAGSystem이 특정 의도를 받도록 보강 가능하지만, 여기서는 자연스러운 흐름 측정
                result = await rag.aquery(query, llm=llm)
                duration = time.time() - start_time
                
                # 결과 기록
                metrics = result.get("performance", {})
                all_results.append({
                    "intent": intent,
                    "actual_intent": result.get("route_decision"),
                    "multiplier": multiplier,
                    "target_k": test_k,
                    "target_top_k": test_top_k,
                    "doc_count": len(result.get("documents", [])),
                    "duration": duration,
                    "tps": metrics.get("tps", 0),
                    "answer_len": len(result.get("response", ""))
                })
                
                print(f"   Result: Docs={len(result.get('documents', []))} | Time={duration:.2f}s | Intent={result.get('route_decision')}")
                
            finally:
                # 원복
                base_params["retrieval_k"] = original_k
                base_params["rerank_top_k"] = original_top_k

    # 결과 출력 및 저장
    df = pd.DataFrame(all_results)
    print("\n=== Evaluation Summary ===")
    print(df.groupby(["intent", "multiplier"])[["doc_count", "duration", "tps", "answer_len"]].mean())
    
    df.to_csv("reports/intent_param_evaluation.csv", index=False)
    print("\nDetailed results saved to 'reports/intent_param_evaluation.csv'")

if __name__ == "__main__":
    asyncio.run(evaluate_params())
