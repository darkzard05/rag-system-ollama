import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from core.reranker import FlashReranker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_reranking():
    # 1. Load PDF
    pdf_path = Path(__file__).parent.parent / "tests" / "data" / "2201.07520v1.pdf"
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading PDF: {pdf_path.name}...")
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Split Documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 3. Setup Retrieval (BM25 as base)
    print("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 30  # Retrieve top 30 candidates

    # 4. Define Query (Korean)
    query = "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야?"
    print(f"\nQuery: {query}")

    # 5. Retrieve Candidates
    print("Retrieving candidates with BM25...")
    candidates = bm25_retriever.invoke(query)
    print(f"Retrieved {len(candidates)} candidates.")

    # 6. Rerank with FlashRank
    print("Initializing FlashReranker (Multilingual)...")
    reranker = FlashReranker()
    
    if not reranker.ranker:
        print("❌ Failed to initialize FlashRank ranker.")
        return

    print("Reranking candidates...")
    reranked_docs = reranker.rerank_documents(query, candidates, top_k=5)

    # 7. Display Results
    print("\n[Top 5 Reranked Results]")
    print("="*80)
    for i, doc in enumerate(reranked_docs):
        score = doc.metadata.get("rerank_score", 0.0)
        content = doc.page_content.replace('\n', ' ')[:200] + "..."
        print(f"Rank {i+1} (Score: {score:.4f}):")
        print(f"Content: {content}")
        print("-" * 80)

        # Check for key terms in the top result
        if i == 0:
            key_terms = ["VQVAE", "GAN", "token", "image", "mask"]
            found_terms = [term for term in key_terms if term.lower() in doc.page_content.lower()]
            if found_terms:
                print(f"✅ Key terms found in top result: {found_terms}")
            else:
                print(f"⚠️ No key terms found in top result. Check content relevance.")

if __name__ == "__main__":
    test_pdf_reranking()
