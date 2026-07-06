import time
import logging
import sys
import os
from pathlib import Path
import psutil
import fitz  # PyMuPDF

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_execution(func, *args, **kwargs):
    """실행 시간 및 메모리 변화를 측정합니다."""
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    return result, (end_time - start_time) * 1000, end_mem - start_mem

def test_hydration_strategies(pdf_path):
    """기존 방식과 정밀 추출 방식을 비교합니다."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return

    logger.info(f"=== 벤치마크 시작: {os.path.basename(pdf_path)} ===")
    
    doc = fitz.open(pdf_path)
    page_num = 0  # 첫 페이지 테스트
    page = doc[page_num]
    
    # 가상의 청크 영역 (페이지 중앙의 작은 영역)
    # 보통 논문의 한 문단 크기 [x0, y0, x1, y1]
    chunk_bbox = fitz.Rect(50, 100, 550, 250) 
    
    # --- 1. 기존 방식: 페이지 전체 단어 추출 ---
    def legacy_extraction():
        words = page.get_text("words")
        return [(w[0], w[1], w[2], w[3], w[4]) for w in words]

    # --- 2. 새로운 방식: 정밀 구역(Clip) 추출 ---
    def precision_extraction():
        # 해당 청크의 영역(clip)만 핀포인트로 읽음
        words = page.get_text("words", clip=chunk_bbox)
        return [(w[0], w[1], w[2], w[3], w[4]) for w in words]

    # 워밍업 (캐시 영향 배제)
    legacy_extraction()
    precision_extraction()

    # 반복 측정 (10회 평균)
    legacy_times = []
    precision_times = []
    
    for _ in range(10):
        _, t_l, _ = measure_execution(legacy_extraction)
        _, t_p, _ = measure_execution(precision_extraction)
        legacy_times.append(t_l)
        precision_times.append(t_p)
        
    res_l = legacy_extraction()
    res_p = precision_extraction()
    
    logger.info(f"\n[기존 방식 - Full Page]")
    logger.info(f"- 평균 소요 시간: {sum(legacy_times)/10: .2f} ms")
    logger.info(f"- 추출된 단어 수: {len(res_l)} 개")
    
    logger.info(f"\n[새로운 방식 - Precision Clip]")
    logger.info(f"- 평균 소요 시간: {sum(precision_times)/10: .2f} ms")
    logger.info(f"- 추출된 단어 수: {len(res_p)} 개")
    
    improvement = (sum(legacy_times) - sum(precision_times)) / sum(legacy_times) * 100
    logger.info(f"\n🚀 성능 개선율: {improvement:.1f}%")
    
    doc.close()

if __name__ == "__main__":
    PDF_FILE = "tests/data/2201.07520v1.pdf"
    test_hydration_strategies(PDF_FILE)
