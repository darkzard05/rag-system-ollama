
import asyncio
import time
import os
from typing import Any
from unittest.mock import MagicMock

# 모듈 경로 추가
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from ui.components.chat import _stream_chat_response, _clean_response_redundancy, normalize_latex_delimiters
from common.utils import apply_tooltips_to_response

async def bench_improved_rendering():
    print("UI Rendering Improved Measurement Start...")
    
    # 가상의 데이터 준비 (텍스트 길이를 더 늘려 차이를 명확히 함)
    full_text = "이것은 매우 긴 테스트 답변입니다. " * 500  # 약 10,000자
    docs = [{"metadata": {"page": i}, "page_content": f"내용 {i}"} for i in range(1, 11)]
    
    # 1. 개선 전 방식 (매번 모든 전처리 수행)
    print(f"Testing current approach with {len(full_text)} chars and {len(docs)} docs...")
    start_time = time.perf_counter()
    for _ in range(20):
        _ = _clean_response_redundancy(full_text)
        _ = normalize_latex_delimiters(full_text)
        if docs:
            _ = apply_tooltips_to_response(full_text, docs)
    duration_old = (time.perf_counter() - start_time) / 20 * 1000
    print(f"Average Preprocessing Time (Current): {duration_old:.2f} ms")
    
    # 2. 개선 후 방식 (스트리밍 중에는 최소 가공만)
    print(f"Testing improved approach (Fast Path) with {len(full_text)} chars...")
    start_time = time.perf_counter()
    for _ in range(20):
        # 스트리밍 중 상황 (Fast Path)
        _ = normalize_latex_delimiters(full_text)
    duration_new = (time.perf_counter() - start_time) / 20 * 1000
    print(f"Average Preprocessing Time (Improved): {duration_new:.2f} ms")
    
    improvement = (duration_old - duration_new) / duration_old * 1000
    print(f"Performance Gain: {((duration_old - duration_new) / duration_old * 100):.1f}%")

    with open("logs/ui_render_perf_comparison.md", "w", encoding="utf-8") as f:
        f.write("# UI Rendering Performance Comparison\n\n")
        f.write(f"- **Scenario**: {len(full_text)} chars, {len(docs)} docs, 20 renders\n")
        f.write(f"- **Baseline (Old)**: {duration_old:.2f} ms\n")
        f.write(f"- **Improved (New)**: {duration_new:.2f} ms\n")
        f.write(f"- **Improvement**: {((duration_old - duration_new) / duration_old * 100):.1f}%\n")

if __name__ == "__main__":
    asyncio.run(bench_improved_rendering())
