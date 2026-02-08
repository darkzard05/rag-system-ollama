import os
import time
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
MARKER_PATH = r"C:\ProgramData\anaconda3\envs\rag-ollama\Scripts\marker_single.exe"

def test_pymupdf4llm():
    import pymupdf4llm
    print("[*] Running PyMuPDF4LLM...")
    s_t = time.time()
    md = pymupdf4llm.to_markdown(TEST_PDF, write_images=False, graphics_limit=100)
    dur = time.time() - s_t
    return md, dur

def test_markitdown():
    from markitdown import MarkItDown
    print("[*] Running MarkItDown...")
    s_t = time.time()
    mid = MarkItDown()
    res = mid.convert(TEST_PDF)
    md = res.text_content
    dur = time.time() - s_t
    return md, dur

def test_marker_cli():
    print("[*] Running Marker (CLI)...")
    out_dir = os.path.join(ROOT_DIR, "logs", "marker_out")
    os.makedirs(out_dir, exist_ok=True)
    
    s_t = time.time()
    # marker_single <file> <out_folder> 형식으로 호출
    try:
        # Note: Marker CLI는 모델 다운로드 등으로 인해 시간이 매우 길어질 수 있음
        subprocess.run([MARKER_PATH, TEST_PDF, out_dir], check=True, capture_output=True)
        dur = time.time() - s_t
        
        # 출력 파일 찾기 (기본적으로 파일명 폴더 생성)
        md_file = os.path.join(out_dir, "2201.07520v1", "2201.07520v1.md")
        if os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                md = f.read()
            return md, dur
    except Exception as e:
        print("[!] Marker CLI Error: " + str(e))
    return "", 0

def compare():
    md_p, t_p = test_pymupdf4llm()
    md_m, t_m = test_markitdown()
    md_k, t_k = test_marker_cli()
    
    print("-" * 50)
    print("Engine | Time | Chars | Table")
    print("-" * 50)
    
    def info(name, md, t):
        if t == 0: return
        has_table = "|" in md[2000:15000]
        line = str(name) + " | " + str(round(t, 2)) + " | " + str(len(md)) + " | " + str(has_table)
        print(line)

    info("PyMuPDF4LLM", md_p, t_p)
    info("MarkItDown", md_m, t_m)
    info("Marker", md_k, t_k)
    print("-" * 50)

if __name__ == "__main__":
    compare()
