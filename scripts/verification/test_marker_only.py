import os
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def test_marker():
    print("[*] Running Marker (AI-based)...")
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        from marker.settings import settings
        
        # 모델 로드 (첫 실행 시 수 분 소요 가능)
        print("    - Loading models...")
        model_lst = load_all_models()
        
        start_time = time.time()
        print("    - Converting...")
        # out_folder가 None이면 메모리에서 처리
        full_text, _, _ = convert_single_pdf(TEST_PDF, model_lst)
        duration = time.time() - start_time
        return full_text, duration
    except Exception as e:
        import traceback
        print("[!] Marker Failed Details:")
        traceback.print_exc()
        return "", 0

if __name__ == "__main__":
    md, t = test_marker()
    if t > 0:
        print("Marker Success! Time: " + str(round(t, 2)) + "s")
        print("Chars: " + str(len(md)))
        print("Table Check: " + str("|" in md[2000:15000]))

