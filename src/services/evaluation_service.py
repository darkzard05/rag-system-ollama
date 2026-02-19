import logging
import time
from datetime import datetime

from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# [주의] FaithfulnesswithHHEM은 v0.4.3 기준 collections에 누락되어 있어 직접 임포트
from ragas.metrics import FaithfulnesswithHHEM
from ragas.metrics.collections import AnswerRelevancy

from common.config import (
    EVAL_JUDGE_MODEL,
    EVAL_TIMEOUT,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
)
from core.model_loader import load_embedding_model

logger = logging.getLogger(__name__)


class EvaluationService:
    """RAG 시스템의 품질을 평가하는 서비스 클래스 (Ragas 0.4+ 균형 모드)"""

    def __init__(self):
        self.report_dir = PROJECT_ROOT / "reports"
        self.report_dir.mkdir(exist_ok=True)

    def _setup_metrics(self):
        """평가 지표 초기화 (균형 모드: 할루시네이션 + 답변 관련성)"""
        # [최적화] 평가용 LLM 설정 (메모리 점유율과 컨텍스트의 균형)
        eval_llm = ChatOllama(
            model=EVAL_JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=EVAL_TIMEOUT,
            temperature=0,
            num_ctx=6144,  # 4096 -> 6144로 확장하여 더 많은 근거 확인 가능
        )

        evaluator_llm = LangchainLLMWrapper(eval_llm)
        embedder = load_embedding_model()
        evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

        # [최적화] FaithfulnesswithHHEM: LLM 호출을 최소화하는 단일 지표
        faithfulness = FaithfulnesswithHHEM(llm=evaluator_llm)

        # [최적화] AnswerRelevancy: strictness=1로 자원 효율적 관련성 평가
        answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1
        )

        logger.info(
            "Successfully initialized balanced evaluation metrics (HHEM + Relevancy)."
        )
        return [faithfulness, answer_relevancy]

    async def run_evaluation(
        self, data_points: list[dict], report_prefix: str = "e2e_eval_report"
    ):
        """
        데이터셋에 대해 균형 잡힌 평가를 수행합니다.
        """
        logger.info(f"Starting balanced evaluation for {len(data_points)} cases...")

        # 1. 데이터셋 구성
        eval_data = []
        for d in data_points:
            raw_context = d.get("context", "")

            # [최적화] 컨텍스트 범위 확장 (상위 5개 청크로 증설하여 정확도 향상)
            if isinstance(raw_context, str) and "### [자료" in raw_context:
                chunks = [
                    c.strip() for c in raw_context.split("### [자료") if c.strip()
                ]
                processed_contexts = [f"### [자료 {c}" for c in chunks[:5]]
            elif isinstance(raw_context, list):
                processed_contexts = raw_context[:5]
            else:
                processed_contexts = [str(raw_context)[:3000]]

            eval_data.append(
                {
                    "user_input": d["query"],
                    "response": d["response"],
                    "retrieved_contexts": processed_contexts,
                }
            )

        dataset = EvaluationDataset.from_list(eval_data)
        metrics = self._setup_metrics()

        # 2. 평가 실행
        start_time = time.time()
        # [안정성] max_workers=1로 순차 실행 유지하여 시스템 멈춤 방지
        run_config = RunConfig(timeout=EVAL_TIMEOUT, max_workers=1)

        results = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

        duration = time.time() - start_time
        logger.info(f"Evaluation finished in {duration:.2f}s")

        # 3. 리포트 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"{report_prefix}_{timestamp}.md"

        summary = results.to_pandas().mean(numeric_only=True).to_dict()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(
                f"# RAG Evaluation Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            )
            f.write(f"**Judge Model:** {EVAL_JUDGE_MODEL}\n")
            f.write(
                "**Optimization:** Balanced Mode (HHEM + Relevancy, Sequential, context-5)\n"
            )
            f.write(f"**Duration:** {duration:.2f}s\n\n")

            f.write("## 📊 Summary Scores\n\n")
            for m, s in summary.items():
                f.write(f"- **{m}:** {s:.4f}\n")

            f.write("\n## 🔍 Detailed Analysis\n\n")
            f.write(results.to_pandas().to_markdown(index=False))

        logger.info(f"Report saved to: {report_path}")
        return summary, str(report_path)
