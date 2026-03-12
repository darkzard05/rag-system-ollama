import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.document_processor import load_pdf_docs
from core.chunking import split_documents
from core.model_loader import ModelManager
from common.config import DEFAULT_EMBEDDING_MODEL

async def verify_sections():
    print("\n" + "=" * 50)
    print("Section Metadata Extraction Verification")
    print("=" * 50)

    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return

    # 1. Load Docs
    print(f"\n1. Loading PDF: {os.path.basename(test_pdf)}")
    docs = load_pdf_docs(test_pdf, os.path.basename(test_pdf))
    
    # 2. Prepare Embedder
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    
    # 3. Semantic Chunking
    print("\n2. Performing Semantic Chunking...")
    chunk_docs, _ = await split_documents(docs, embedder)
    
    # 4. Analyze unique sections
    sections = []
    seen = set()
    for d in chunk_docs:
        sec = d.metadata.get("current_section", "N/A")
        if sec not in seen:
            sections.append(sec)
            seen.add(sec)
    
    print("\n3. Extracted Unique Sections:")
    for i, s in enumerate(sections):
        print(f"   [{i+1}] {s}")
        
    # '3 CM3' 뒤에 불필요한 텍스트가 붙어있는지 확인
    cm3_section = next((s for s in sections if "3 CM3" in s), None)
    if cm3_section:
        print(f"\nTarget Check ('3 CM3'):")
        print(f"   Actual: '{cm3_section}'")
        if cm3_section.strip() == "3 CM3":
            print("   ✅ SUCCESS: Section title is clean!")
        else:
            print("   ⚠️ WARNING: Section title still contains extra text.")
    else:
        print("\n❌ ERROR: '3 CM3' section not found.")

if __name__ == "__main__":
    asyncio.run(verify_sections())
