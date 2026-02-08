import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def test_pymupdf4llm():
    import pymupdf4llm
    print("[*] Testing PyMuPDF4LLM...")
    start_time = time.time()
    md_text = pymupdf4llm.to_markdown(TEST_PDF, write_images=False, graphics_limit=100)
    duration = time.time() - start_time
    char_count = len(md_text)
    print("    - Time: " + str(round(duration, 2)) + "s")
    print("    - Chars: " + str(char_count))
    return md_text, duration

def test_docling():
    from docling.document_converter import DocumentConverter
    print("[*] Testing Docling (IBM)...")
    start_time = time.time()
    converter = DocumentConverter()
    result = converter.convert(TEST_PDF)
    md_text = result.document.export_to_markdown()
    duration = time.time() - start_time
    char_count = len(md_text)
    print("    - Time: " + str(round(duration, 2)) + "s")
    print("    - Chars: " + str(char_count))
    return md_text, duration

def compare():
    base_name = os.path.basename(TEST_PDF)
    print("Benchmark File: " + base_name)
    print("")
    
    md_p, time_p = test_pymupdf4llm()
    md_d, time_d = test_docling()
    
    print("-" * 60)
    print("Metric               | PyMuPDF4LLM     | Docling")
    print("-" * 60)
    
    line_time = "Time (Seconds)       | " + str(round(time_p, 2)) + "            | " + str(round(time_d, 2))
    print(line_time)
    
    ratio = round(time_d / time_p, 2) if time_p > 0 else 0
    line_ratio = "Speed Ratio          | 1.0x            | " + str(ratio) + "x"
    print(line_ratio)
    
    line_chars = "Char Count           | " + str(len(md_p)) + "           | " + str(len(md_d))
    print(line_chars)
    
    print("\n[Sampling: Abstract Area]")
    print("-" * 30)
    
    idx_p = md_p.find("ABSTRACT")
    if idx_p != -1:
        print("PYMUPDF: " + md_p[idx_p:idx_p+150])
    
    idx_d = md_d.find("ABSTRACT")
    if idx_d != -1:
        print("DOCLING: " + md_d[idx_d:idx_d+150])

    has_table_p = "|" in md_p[5000:15000]
    has_table_d = "|" in md_d[5000:15000]
    print("\nTable Detected: PYMUPDF=" + str(has_table_p) + ", DOCLING=" + str(has_table_d))

if __name__ == "__main__":
    compare()