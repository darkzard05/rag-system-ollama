import time
import pymupdf4llm
from pathlib import Path

def benchmark_extraction(file_path: str):
    print(f"[*] Analyzing Document: {file_path}")
    
    # 1. Old Method (Graphics Limit 0)
    start = time.perf_counter()
    old_md = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
        graphics_limit=0,
        table_strategy="lines",
        show_progress=False
    )
    old_time = time.perf_counter() - start
    
    # 2. New Method (Graphics Limit 500)
    start = time.perf_counter()
    new_md = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
        graphics_limit=500,
        table_strategy="lines_strict",
        show_progress=False
    )
    new_time = time.perf_counter() - start
    
    # Stats
    old_chars = sum(len(c.get("text", "")) for c in old_md)
    new_chars = sum(len(c.get("text", "")) for c in new_md)
    
    print("=" * 60)
    print("Metric               | Old (Limit 0)   | New (Limit 500)")
    print("-" * 60)
    print(f"Execution Time       | {old_time:.4f}s      | {new_time:.4f}s")
    print(f"Total Chunks         | {len(old_md)}             | {len(new_md)}")
    print(f"Total Characters     | {old_chars}          | {new_chars}")
    
    char_diff = new_chars - old_chars
    print(f"Char Difference      | {char_diff}")
    
    table_pattern = "|"
    old_tab_count = sum(1 for c in old_md if table_pattern in c.get("text", ""))
    new_tab_count = sum(1 for c in new_md if table_pattern in c.get("text", ""))
    print(f"Chunks with '|'      | {old_tab_count}             | {new_tab_count}")
    print("=" * 60)

if __name__ == "__main__":
    test_file = "tests/data/2201.07520v1.pdf"
    if Path(test_file).exists():
        benchmark_extraction(test_file)
    else:
        print("File not found")
