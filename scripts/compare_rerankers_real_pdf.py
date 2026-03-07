
import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Force CPU to avoid device mismatch errors in this environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["EMBEDDING_DEVICE"] = "cpu"

from core.model_loader import ModelManager
from core.reranker import DistributedReranker, RerankerStrategy
from core.document_processor import load_pdf_docs
from core.retriever_factory import create_vector_store
from api.schemas import AggregatedSearchResult

async def run_benchmark():
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    query = "What are the main findings of the paper regarding chain-of-thought prompting?"
    
    print(f"--- RAG Pipeline Setup (PDF: {os.path.basename(pdf_path)}) ---")
    
    # 1. Load Embedder explicitly on CPU
    print("1. Loading Embedding Model (CPU)...")
    embedder = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"}
    )
    
    # 2. Load and Split Documents
    print("2. Loading and Splitting PDF...")
    # Use standard splitter to avoid SemanticChunker complexity/errors
    raw_docs = load_pdf_docs(pdf_path, os.path.basename(pdf_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    splits = text_splitter.split_documents(raw_docs)
    print(f"   Created {len(splits)} chunks.")
    
    # 3. Build Vector Store
    print("3. Building FAISS Index...")
    vector_store = create_vector_store(splits, embedder)
    
    # 4. Retrieval
    print(f"4. Retrieving Candidates for query: '{query}'")
    # Search for top 20 candidates
    # Using similarity_search_with_relevance_scores to get scores (0 to 1 range for cosine)
    results_with_scores = await vector_store.asimilarity_search_with_relevance_scores(query, k=20)
    
    candidates = []
    for doc, score in results_with_scores:
        candidates.append(AggregatedSearchResult(
            doc_id=doc.metadata.get("doc_id", str(hash(doc.page_content))),
            content=doc.page_content,
            score=float(score),
            node_id="faiss",
            metadata=doc.metadata
        ))
    
    print(f"   Retrieved {len(candidates)} candidates.")

    # 5. Baseline: FlashRank (Current Strategy)
    print("\n--- Strategy 1: FlashRank (Current Baseline) ---")
    ranker_flash = await ModelManager.get_flashranker()
    from flashrank import RerankRequest
    passages = [{"id": r.doc_id, "text": r.content, "meta": r.metadata} for r in candidates]
    
    start_time = time.time()
    flash_results = ranker_flash.rerank(RerankRequest(query=query, passages=passages))
    flash_latency = time.time() - start_time
    
    flash_docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in flash_results[:5]]
    
    # 6. Proposed: DistributedReranker with DIVERSITY (Proposed Strategy)
    print("\n--- Strategy 2: DistributedReranker (DIVERSITY/MMR) ---")
    reranker_dist = DistributedReranker()
    
    start_time = time.time()
    # diversity_weight 0.7 for strong diversity
    dist_results, metrics = reranker_dist.rerank(
        results=candidates,
        query_text=query,
        strategy=RerankerStrategy.DIVERSITY,
        top_k=5,
        diversity_weight=0.7
    )
    dist_latency = time.time() - start_time
    
    dist_docs = [Document(page_content=r.content, metadata=r.metadata) for r in dist_results]

    # 7. Comparison and Evaluation
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'FlashRank (Baseline)':<20} | {'DIVERSITY (MMR)'}")
    print("-" * 80)
    
    def get_diversity_stats(docs):
        pages = [d.metadata.get("page") for d in docs]
        unique_pages = len(set(pages))
        return unique_pages, pages

    flash_unique, flash_pages = get_diversity_stats(flash_docs)
    dist_unique, dist_pages = get_diversity_stats(dist_docs)
    
    print(f"{'Latency (sec)':<25} | {flash_latency:<20.4f} | {dist_latency:.4f}")
    print(f"{'Unique Pages (Count)':<25} | {flash_unique:<20} | {dist_unique}")
    print(f"{'Page Distribution':<25} | {str(flash_pages):<20} | {str(dist_pages)}")
    print("="*80)

    print("\n[FlashRank - Top 3 Contexts]")
    for i, d in enumerate(flash_docs[:3]):
        print(f" {i+1}. [Page {d.metadata.get('page')}] {d.page_content[:150].strip()}...")

    print("\n[DIVERSITY - Top 3 Contexts]")
    for i, d in enumerate(dist_docs[:3]):
        print(f" {i+1}. [Page {d.metadata.get('page')}] {d.page_content[:150].strip()}...")

    print("\nConclusion:")
    if dist_unique > flash_unique:
        print(">> The Proposed DIVERSITY strategy successfully gathered information from more diverse pages.")
    elif dist_unique == flash_unique:
        print(">> Both strategies resulted in the same number of unique pages.")
    else:
        print(">> FlashRank resulted in more unique pages.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
