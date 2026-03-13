
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

async def bench_current_rendering():
    print("UI Rendering Baseline Measurement Start...")
    
    # 가상의 데이터 준비
    full_text = "이것은 긴 테스트 답변입니다. " * 200  # 약 4000자
    docs = [{"metadata": {"page": i}, "page_content": f"내용 {i}"} for i in range(1, 11)]
    
    # 1. 현재 방식 (매번 모든 전처리 수행)
    print(f"Testing current approach with {len(full_text)} chars and {len(docs)} docs...")
    
    start_time = time.perf_counter()
    for _ in range(20):  # 20회 반복 측정 (0.05초 주기로 1초간 렌더링 상황 가정)
        # 0.05초 주기 흉내
        display_text = _clean_response_redundancy(full_text)
        display_text = normalize_latex_delimiters(display_text)
        if docs:
            display_text = apply_tooltips_to_response(display_text, docs)
        # st.markdown()은 생략 (순수 전처리 시간 측정)
        
    duration = (time.perf_counter() - start_time) / 20 * 1000
    print(f"Average Preprocessing Time (Current): {duration:.2f} ms")
    
    with open("logs/ui_render_perf.csv", "a", encoding="utf-8") as f:
        f.write(f"{len(full_text)},{duration:.2f}\n")

if __name__ == "__main__":
    asyncio.run(bench_current_rendering())
