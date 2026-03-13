import asyncio
import time
import os
import sys
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

import pymupdf4llm
from core.document_processor import load_pdf_docs

def analyze_table_quality(chunks: List[Dict[str, Any]]):
    """청크 리스트에서 표 데이터를 추출하여 정밀 통계를 산출합니다."""
    total_tables = 0
    total_rows = 0
    total_cols = 0
    
    for chunk in chunks:
        tables = chunk.get("tables", [])
        total_tables += len(tables)
        for t in tables:
            total_rows += t.get("rows", 0)
            total_cols += t.get("columns", 0)
            
    total_text = "".join([c.get("text", "") for c in chunks])
    marker_count = total_text.count("|---|") + total_text.count("| --- |")
    
    # 품질 지수 계산 (실제 추출된 표 구조와 마크다운 마커의 일치도)
    quality_score = 0
    if total_tables > 0:
        # 추출된 표 개수 대비 마크다운 내 표 마커 비중 (1.0에 가까울수록 구조적 일치도 높음)
        quality_score = min(100, (marker_count / total_tables) * 100 if total_tables > 0 else 0)

    return {
        "table_count": total_tables,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "marker_count": marker_count,
        "quality_score": round(quality_score, 1),
        "text_length": len(total_text),
    }

def run_systematic_benchmark(file_path: str):
    print(f"\n[Systematic Table Extraction Benchmark] 시작: {Path(file_path).name}")
    print("-" * 80)
    
    # 테스트 시나리오 정의
    scenarios = [
        {"id": "V1", "name": "Quality-First", "strategy": "lines_strict", "limit": None, "ignore": False},
        {"id": "V2", "name": "Standard-Balanced", "strategy": "lines", "limit": 5000, "ignore": False},
        {"id": "V3", "name": "Mid-Performance", "strategy": "lines", "limit": 2000, "ignore": False},
        {"id": "V4", "name": "Fast-Lossy (Text)", "strategy": "text", "limit": 500, "ignore": True},
        {"id": "V5", "name": "Strict-Limited", "strategy": "lines_strict", "limit": 1000, "ignore": False},
    ]
    
    results = []
    
    for sc in scenarios:
        print(f"[*] [{sc['id']}] {sc['name']} 실행 중... (strategy={sc['strategy']}, limit={sc['limit']})")
        start = time.perf_counter()
        try:
            chunks = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                ignore_graphics=sc['ignore'],
                graphics_limit=sc['limit'],
                table_strategy=sc['strategy']
            )
            elapsed = time.perf_counter() - start
            stats = analyze_table_quality(chunks)
            
            # 종합 점수 (품질 70% + 속도 30%)
            # 기준 속도는 V2(Standard)를 100%로 잡음 (대략 7초)
            speed_score = (7.0 / elapsed) * 100 if elapsed > 0 else 0
            final_score = (stats['quality_score'] * 0.7) + (min(100, speed_score) * 0.3)
            
            stats.update({
                "ID": sc['id'],
                "Name": sc['name'],
                "Strategy": sc['strategy'],
                "Limit": str(sc['limit']),
                "Time (s)": round(elapsed, 2),
                "Tables": stats['table_count'],
                "Rows": stats['total_rows'],
                "Quality (%)": stats['quality_score'],
                "Efficiency Score": round(final_score, 1)
            })
            results.append(stats)
        except Exception as e:
            print(f"[!] [{sc['id']}] 실패: {e}")

    # 결과 분석 및 순위 산정
    df = pd.DataFrame(results)
    df = df.sort_values(by="Efficiency Score", ascending=False)
    
    print("\n" + "="*100)
    cols = ["ID", "Name", "Strategy", "Time (s)", "Tables", "Rows", "Quality (%)", "Efficiency Score"]
    print(df[cols].to_string(index=False))
    print("="*100)
    
    # 최적 설정 추천
    best = df.iloc[0]
    print(f"\n[Recommendation]")
    print(f"✓ 최적 설정: {best['Name']} ({best['ID']})")
    print(f"  - 이유: 품질 {best['Quality (%)']}% 확보 및 {best['Time (s)']}초의 안정적인 속도")
    print(f"  - 권장 table_strategy: '{best['Strategy']}'")
    print(f"  - 권장 graphics_limit: {best['Limit']}")

    return results

if __name__ == "__main__":
    test_file = "tests/data/2201.07520v1.pdf"
    if Path(test_file).exists():
        run_systematic_benchmark(test_file)
    else:
        print("테스트 PDF 파일을 찾을 수 없습니다.")
