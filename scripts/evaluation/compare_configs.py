
import asyncio
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib

# 프로젝트 루트 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(ROOT_DIR / "scripts"))

from evaluation.quick_eval import run_quick_evaluation

def update_config(bm25_w, faiss_w, rerank_enabled):
    """config.yml 파일을 동적으로 업데이트합니다."""
    config_path = ROOT_DIR / "config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 가중치 업데이트
    if 'rag' in cfg and 'retriever' in cfg['rag']:
        cfg['rag']['retriever']['ensemble_weights'] = [bm25_w, faiss_w]
    
    # 리랭킹 업데이트
    if 'rag' in cfg and 'reranker' in cfg['rag']:
        cfg['rag']['reranker']['enabled'] = rerank_enabled
        
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    
    # 설정 재로드를 위해 common.config 강제 리로드
    if 'common.config' in sys.modules:
        importlib.reload(sys.modules['common.config'])
    
    print("\n[Config] BM25:" + str(bm25_w) + ", FAISS:" + str(faiss_w) + ", Rerank:" + str(rerank_enabled) + " 설정 적용 완료")

async def run_comparison():
    pdf = "tests/data/2201.07520v1.pdf"
    csv = "tests/data/testset_2201.csv"
    
    test_cases = [
        {"id": "Default", "bm25": 0.4, "faiss": 0.6, "rerank": True},
        {"id": "Semantic+", "bm25": 0.2, "faiss": 0.8, "rerank": True},
        {"id": "Keyword+", "bm25": 0.6, "faiss": 0.4, "rerank": True},
        {"id": "No-Rerank", "bm25": 0.4, "faiss": 0.6, "rerank": False},
    ]
    
    all_summaries = []
    
    # 원본 설정 백업
    config_path = ROOT_DIR / "config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        original_cfg = f.read()

    try:
        for case in test_cases:
            print("\n=== 실험 시작: " + case['id'] + " ===")
            update_config(case['bm25'], case['faiss'], case['rerank'])
            
            try:
                # 결과 수집 방식을 파일 검색이 아닌 리턴값 직접 수령으로 변경 유도 (또는 최신 파일 정확히 특정)
                await run_quick_evaluation(pdf, csv)
                
                # reports 폴더에서 가장 최신 quick_eval CSV 읽기 (수정된 로직)
                report_dir = ROOT_DIR / "reports"
                csv_files = list(report_dir.glob("quick_eval_*.csv"))
                if csv_files:
                    latest_report = max(csv_files, key=os.path.getmtime)
                    res_df = pd.read_csv(latest_report)
                    avg_score = res_df['score'].mean()
                    avg_sim = res_df['similarity'].mean()
                else:
                    avg_score, avg_sim = 0.0, 0.0
                
                summary = {
                    "Case": case['id'],
                    "Avg_Score": round(avg_score, 2),
                    "Avg_Sim": round(avg_sim, 4)
                }
                all_summaries.append(summary)
                print(">>> " + case['id'] + " 결과: Score " + str(summary['Avg_Score']))
            except Exception as e:
                print("실험 중 오류 발생: " + str(e))

    finally:
        # 원본 설정 복구
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(original_cfg)
        print("\n[System] 원본 설정 복구 완료")

    # 최종 결과 출력
    if all_summaries:
        final_df = pd.DataFrame(all_summaries)
        print("\n" + "="*50)
        print("설정별 비교 결과 요약")
        print("="*50)
        print(final_df.to_string(index=False))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_df.to_csv(ROOT_DIR / "reports" / ("compare_results_" + timestamp + ".csv"), index=False)
    else:
        print("\n[Error] 수집된 실험 결과가 없습니다.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
