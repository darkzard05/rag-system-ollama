import time
import pymupdf4llm
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def test_config(name, **kwargs):
    print(f"[*] Testing: {name}")
    start = time.time()
    try:
        md = pymupdf4llm.to_markdown(TEST_PDF, **kwargs)
        dur = time.time() - start
        print(f"    - Time: {dur:.2f}s")
        return dur
    except Exception as e:
        print(f"    - Failed: {e}")
        return None

def run_bench():
    results = {}
    results['Default'] = test_config("Default", write_images=False)
    results['Limit Graphics 100'] = test_config("Limit Graphics 100", write_images=False, graphics_limit=100)
    results['No Graphics 0'] = test_config("No Graphics 0", write_images=False, graphics_limit=0)
    
    print("\n[Summary]")
    for k, v in results.items():
        if v:
            line = f"{k}: {v:.2f}s"
            print(line)

if __name__ == "__main__":
    run_bench()