import time
import logging
import sys
import os
from pathlib import Path
from flashrank import Ranker, RerankRequest

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def measure_rerank(model_name, query, passages):
    try:
        # 모델 로드
        ranker = Ranker(model_name=model_name, cache_dir=".model_cache")
        
        # 측정 시작
        start_time = time.perf_counter()
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000
        top_result_id = results[0]["id"]
        target_rank = next((i + 1 for i, r in enumerate(results) if r["id"] == 0), -1)
        top_score = results[0]["score"]
        
        return {
            "model": model_name,
            "latency_ms": latency,
            "target_rank": target_rank,
            "top_score": top_score,
            "is_correct": top_result_id == 0
        }
    except Exception as e:
        logger.error(f"모델 {model_name} 실행 실패: {e}")
        return None

def run_benchmark_v2():
    """더 복잡한 질문과 다양한 모델로 2차 벤치마크를 수행합니다."""
    
    # 더 정교한 차이를 요구하는 질문
    query = "DeepSeek-R1의 추론 과정에서의 강화학습(RL) 적용 방식과 성능 수치를 구체적으로 알려주세요."
    
    passages = [
        {
            "id": 0, # 정답 (정교한 정보 포함)
            "text": "DeepSeek-R1은 순수 강화학습(Pure RL)을 통해 추론 능력을 학습시켰으며, 이를 통해 'AIME 2024' 벤치마크에서 79.8%의 높은 점수를 기록했습니다. MoE 구조를 활용하여 효율성을 높였습니다."
        },
        {
            "id": 1, # 오답 (유사한 수치를 가진 다른 모델)
            "text": "일반적인 추론 모델인 GPT-4는 다양한 벤치마크에서 높은 점수를 보이며, 지도 학습(SFT) 기반으로 훈련되었습니다. 강화학습도 사용되지만 R1과는 방식이 다릅니다."
        },
        {
            "id": 2, # 오답 (수치가 틀림)
            "text": "DeepSeek-R1은 지도 학습만으로 개발되었으며, 벤치마크 점수는 약 50% 수준입니다. 이는 기존 모델 대비 낮은 수치입니다."
        },
        {
            "id": 3, # 노이즈
            "text": "강화학습은 인공지능 분야의 핵심 기술 중 하나입니다."
        }
    ]
    
    # 노이즈 추가
    for i in range(4, 25):
        passages.append({"id": i, "text": f"무작위 문서 {i}: 딥러닝과 강화학습의 결합은 현대 AI의 트렌드입니다."})

    # FlashRank 공식 지원 모델들
    models = [
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-12-v2",
        "rank-vicuna-7b-v1-nwp",  # Vicuna 기반 고성능 (시도)
        "rank_zephyr_7b_v1_full"  # Zephyr 기반 (매우 무거울 수 있음)
    ]

    logger.info("=== Reranker Model Benchmark V2 시작 ===")
    
    results = []
    for model in models:
        logger.info(f"테스트 중: {model}...")
        res = measure_rerank(model, query, passages)
        if res:
            results.append(res)

    logger.info("\n" + "="*80)
    logger.info(f"{'Model Name':<30} | {'Latency':<10} | {'Rank-1':<8} | {'Score':<8}")
    logger.info("-" * 80)
    for r in results:
        status = "✅ PASS" if r["is_correct"] else f"❌ FAIL (P{r['target_rank']})"
        logger.info(f"{r['model']:<30} | {r['latency_ms']:>6.2f} ms | {status:<8} | {r['top_score']:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    run_benchmark_v2()
