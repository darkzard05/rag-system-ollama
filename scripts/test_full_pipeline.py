import os
import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL
from common.logging_config import setup_logging

async def run_evaluation(data_points: list[dict]):
    """Ragas를 사용하여 생성된 답변의 품질을 평가합니다."""
    print("\n" + "=" * 50)
    print("[Ragas] Starting Automatic Quality Evaluation...")
    
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama
        from ragas import RunConfig

        # 1. 데이터셋 구성
        eval_data = []
        for d in data_points:
            eval_data.append({
                "user_input": d["query"],
                "response": d["response"],
                "retrieved_contexts": [d["context"]] if d["context"] else ["(No context)"]
            })
        
        dataset = EvaluationDataset.from_list(eval_data)

        # 2. 평가기 설정 (로컬 Ollama 사용, JSON 출력 및 결정론적 추론 강화)
        llm = ChatOllama(
            model=DEFAULT_OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL,
            timeout=600,
            format="json", # [해결책 2] Ollama JSON 모드 강제
            temperature=0,  # [해결책 3] 출력 안정성 극대화
            num_ctx=8192
        )
        evaluator_llm = LangchainLLMWrapper(llm)
        
        embedder = load_embedding_model()
        evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

        # [참고] Ragas는 내부적으로 프롬프트를 처리하지만, 
        # format="json" 설정이 적용된 Ollama는 LLM이 텍스트를 섞어 쓰는 것을 방지합니다.

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        ]

        # 3. 평가 실행
        print(f"[Ragas] Starting evaluation for {len(dataset)} cases...")
        print("[Ragas] Running metrics (Faithfulness, AnswerRelevancy)...")
        
        start_eval_time = time.time()
        run_config = RunConfig(timeout=300, max_workers=1) 
        
        # [최적화] 결과 직접 확인을 위해 evaluate 호출부 감싸기
        results = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
        
        eval_duration = time.time() - start_eval_time
        print(f"[Ragas] Evaluation finished in {eval_duration:.2f}s")

        # 4. 리포트 저장
        report_dir = ROOT_DIR / "reports"
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"e2e_eval_report_{timestamp}.md"

        summary = results.to_pandas().mean(numeric_only=True).to_dict()
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# E2E Pipeline Evaluation Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
            f.write(f"**Evaluator Model:** {DEFAULT_OLLAMA_MODEL}\n\n")
            f.write("## 📊 Summary Scores\n\n")
            for m, s in summary.items():
                f.write(f"- **{m}:** {s:.4f}\n")
            f.write("\n## 🔍 Detailed Analysis\n\n")
            f.write(results.to_pandas().to_markdown(index=False))

        print(f"[Ragas] Evaluation complete. Scores: {summary}")
        print(f"[Ragas] Detailed report saved to: {report_path}")
        print("=" * 50)

    except Exception as e:
        print(f"[Ragas] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

async def run_full_pipeline_test():
    # 로깅 설정 초기화
    setup_logging(log_level="INFO", log_file=ROOT_DIR / "logs" / "test_e2e.log")
    
    print("[E2E] RAG Pipeline Integration Test Started")
    
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    rag = RAGSystem(session_id=session_id)
    
    print("1. Loading Models...")
    try:
        embedder = load_embedding_model()
        llm = load_llm(DEFAULT_OLLAMA_MODEL)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print(f"2. Indexing Document: {os.path.basename(test_pdf)}")
    
    start_time = asyncio.get_event_loop().time()
    msg, cache_used = await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)
    load_time = asyncio.get_event_loop().time() - start_time
    print(f"   Result: {msg} (Cache: {cache_used}) | Time: {load_time:.2f}s")

    # 평가를 위한 테스트 쿼리 세트 (빠른 검증을 위해 1개로 제한)
    test_cases = [
        "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    ]
    
    captured_data = []

    print("\n3. Running Test Queries & Collecting Data...")
    for i, test_query in enumerate(test_cases):
        print(f"   [{i+1}/{len(test_cases)}] Querying: '{test_query[:50]}...'")
        
        start_t = asyncio.get_event_loop().time()
        result = await rag.aquery(test_query, llm=llm)
        q_time = asyncio.get_event_loop().time() - start_t
        
        print(f"   -> Done ({q_time:.2f}s)")
        
        captured_data.append({
            "query": test_query,
            "response": result.get("response", ""),
            "context": result.get("context", "")
        })

    print(f"\n4. Pipeline Test Finished (Total Queries: {len(test_cases)})")
    
    # 5. [추가] 즉시 평가 실행
    await run_evaluation(captured_data)

if __name__ == "__main__":
    asyncio.run(run_full_pipeline_test())