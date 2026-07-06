
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test")

def test_imports():
    try:
        from docling.document_converter import DocumentConverter
        logger.info("✅ Docling import 성공")
    except ImportError as e:
        logger.error(f"❌ Docling import 실패: {e}")

    try:
        from flashrank import Ranker
        logger.info("✅ FlashRank import 성공")
    except ImportError as e:
        logger.error(f"❌ FlashRank import 실패: {e}")

    try:
        from langchain_docling import DoclingLoader
        logger.info("✅ LangChain-Docling import 성공")
    except ImportError as e:
        logger.error(f"❌ LangChain-Docling import 실패: {e}")

if __name__ == "__main__":
    test_imports()
