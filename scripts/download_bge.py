import time
import logging
import sys
import os

# 절대 경로를 사용하여 src 폴더를 경로에 추가
sys.path.append(r"C:\Users\darkzard05\hy\rag-system-ollama\src")

try:
    from core.reranker import BGEReranker

    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model():
    print("Downloading BGE model... This may take a few minutes.")
    try:
        BGEReranker()
        print("✅ Model downloaded and loaded successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    download_model()
