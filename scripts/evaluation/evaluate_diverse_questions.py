import os
import sys
import asyncio
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL

async def run_evaluation():
    print("[Evaluation] Starting Diverse Question Test Suite...")
    
    session_id = "eval-suite-diverse"
    rag = RAGSystem(session_id=session_id)
    
    embedder = load_embedding_model()
    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    
    # 1. 문서 로드
    await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)

    questions = [
        "CM3 모델의 Medium과 Large 버전은 각각 몇 개의 파라미터를 가지고 있나요?",
        "Causally Masked Language Modeling이 양방향 맥락(Bidirectional context)을 제공하는 원리가 무엇인가요?",
        "CM3 모델이 제로샷(Zero-shot)으로 수행할 수 있는 작업들을 모두 나열해 주세요.",
        "Entity Linking 작업에서 CM3 모델은 기존 SOTA와 비교해 어떤 성과를 냈나요?",
        "CM3가 이미지 인필링(Image In-filling)을 수행할 때 사용하는 프롬프트 구조는 어떤 식인가요?"
    ]

    results = []
    for i, q in enumerate(questions):
        print(f"\n[Q{i+1}] {q}")
        start_time = asyncio.get_event_loop().time()
        res = await rag.aquery(q, llm=llm)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        ctx_len = len(res.get('context', ''))
        print(f"   Done ({elapsed:.2f}s) | Context: {ctx_len} chars")
        results.append({
            "question": q,
            "answer": res.get("response"),
            "context": res.get("context")
        })

    # 최종 결과 저장
    output_path = ROOT_DIR / "logs" / "eval" / "diverse_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation Complete. Results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())

