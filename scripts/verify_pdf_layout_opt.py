import time
import fitz
import sys
import os

def benchmark_pdf_extraction(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    doc = fitz.open(file_path)
    total_pages = len(doc)
    print(f"Target PDF: {os.path.basename(file_path)} ({total_pages} pages)")
    print("-" * 60)

    # 1. Original Method: Plain text
    start_time = time.perf_counter()
    original_texts = []
    for page in doc:
        original_texts.append(page.get_text())
    original_duration = (time.perf_counter() - start_time) * 1000
    
    # 2. Optimized Method: Layout-aware blocks
    start_time = time.perf_counter()
    optimized_texts = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        page_text = "\n\n".join([b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()])
        optimized_texts.append(page_text)
    optimized_duration = (time.perf_counter() - start_time) * 1000

    print(f"{'Method':<25} | {'Time (ms)':<12} | {'Avg/Page (ms)':<12}")
    print("-" * 60)
    print(f"{'Original (Plain)':<25} | {original_duration:10.2f} | {original_duration/total_pages:10.2f}")
    print(f"{'Optimized (Blocks+Sort)':<25} | {optimized_duration:10.2f} | {optimized_duration/total_pages:10.2f}")
    print(f"Overhead: {optimized_duration/original_duration:.2f}x")
    print("-" * 60)

    print("\n[Page 1 Snippet - Original (Plain)]")
    print("-" * 30)
    print(original_texts[0][:400].replace('\n', ' | ')[:400] + "...")
    
    print("\n[Page 1 Snippet - Optimized (Blocks+Sort)]")
    print("-" * 30)
    print(optimized_texts[0][:400].replace('\n', ' | ')[:400] + "...")

    doc.close()

if __name__ == '__main__':
    pdf_path = 'tests/data/2201.07520v1.pdf'
    benchmark_pdf_extraction(pdf_path)