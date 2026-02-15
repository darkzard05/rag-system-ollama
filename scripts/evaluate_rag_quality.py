import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 설정
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from langchain_ollama import ChatOllama
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from core.model_loader import load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL

def get_latest_log():
    log_dir = ROOT_DIR / "logs" / "eval"
    logs = list(log_dir.glob("qa_history_*.jsonl"))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)

async def run_ragas_eval():
    print("[Ragas] Starting RAG Quality Evaluation...")
    
    log_path = get_latest_log()
    if not log_path:
        print("Error: No log files found in logs/eval/")
        return

    print(f"1. Loading latest log: {log_path.name}")
    data = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                contexts = [entry.get("context", "")] if entry.get("context") else ["(No context provided)"]
                data.append({
                    "user_input": entry.get("query"),
                    "response": entry.get("response"),
                    "retrieved_contexts": contexts
                })
            except Exception:
                continue

    if not data:
        print("Error: No valid data found in log file.")
        return

    # 최근 3개만 테스트 (시간 단축 및 검증용)
    eval_data = data[-3:]
    dataset = EvaluationDataset.from_list(eval_data)

    print("2. Setting up Local Evaluator (Ollama)...")
    llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    embedder = load_embedding_model()
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

    # 기본 지표 2개만 우선 측정
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]

    print("3. Running Evaluation...")
    from ragas import RunConfig
    run_config = RunConfig(timeout=120, max_workers=1) # 로컬 모델 안정성을 위해 타임아웃 연장 및 순차 처리

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config
    )

    # 4. 리포트 생성
    print("4. Generating Report...")
    report_path = ROOT_DIR / "reports" / f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs(ROOT_DIR / "reports", exist_ok=True)
    
    header = f"# RAG Quality Evaluation Report ({datetime.now().strftime('%Y-%m-%d')})\n\n"
    source = f"**Source Log:** {log_path.name}\n"
    model_info = f"**Evaluator Model:** {DEFAULT_OLLAMA_MODEL}\n\n"
    
    # [수정] 결과 딕셔너리 추출 방식 변경
    final_scores = results.scores
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(source)
        f.write(model_info)
        f.write("## 📊 Summary Scores\n\n")
        
        # 전체 평균 점수 기록
        # results 객체는 내부적으로 딕셔너리처럼 동작하거나 to_pandas() 활용 가능
        summary = results.to_pandas().mean(numeric_only=True).to_dict()
        for metric_name, score in summary.items():
            f.write(f"- **{metric_name}:** {score:.4f}\n")
        
        f.write("\n## 🔍 Detailed Results\n\n")
        df = results.to_pandas()
        f.write(df.to_markdown(index=False))

    print(f"\nEvaluation Complete! Report saved to: {report_path}")
    print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_ragas_eval())
