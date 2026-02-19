import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from langchain_ollama import ChatOllama
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from services.evaluation_service import EvaluationService
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
                    "query": entry.get("query"),
                    "response": entry.get("response"),
                    "context": contexts
                })
            except Exception:
                continue

    if not data:
        print("Error: No valid data found in log file.")
        return

    # 최근 5개만 테스트 (시간 단축 및 검증용)
    eval_data = data[-5:]
    
    print("2. Running Evaluation via EvaluationService...")
    eval_service = EvaluationService()
    summary, report_path = await eval_service.run_evaluation(eval_data, report_prefix="ragas_log_report")

    print(f"\nEvaluation Complete! Report saved to: {report_path}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_ragas_eval())

