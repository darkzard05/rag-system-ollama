import time
import os
import torch
import numpy as np
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmbeddingBenchmark")

# 테스트 데이터 (실제 문장들과 유사하게 구성)
TEST_TEXTS = [
    "인공지능 기술은 현대 사회의 다양한 분야에서 혁신을 일으키고 있습니다.",
    "The transformer architecture has revolutionized natural language processing.",
    "RAG(Retrieval-Augmented Generation) improves the accuracy of large language models.",
    "NumPy is a fundamental package for scientific computing with Python.",
    "Efficient vector search is crucial for high-performance RAG systems."
] * 20  # 총 100개 문장

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = ".model_cache"

def benchmark_standard_pytorch():
    logger.info("--- [Standard PyTorch] Benchmarking ---")
    start_load = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        cache_folder=CACHE_DIR
    )
    load_time = time.time() - start_load
    logger.info(f"Model Load Time: {load_time:.2f}s")
    
    # 워밍업
    embeddings.embed_query("Warmup text")
    
    start_inference = time.time()
    for _ in range(5):  # 5회 반복 측정
        _ = embeddings.embed_documents(TEST_TEXTS)
    inference_time = (time.time() - start_inference) / 5
    
    logger.info(f"Average Inference Time (100 docs): {inference_time:.4f}s")
    return load_time, inference_time

def benchmark_optimum_onnx():
    logger.info("--- [Optimum ONNX CPU] Benchmarking ---")
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("optimum or onnxruntime not installed correctly.")
        return None, None
        
    onnx_path = Path(CACHE_DIR) / "onnx_cpu_v1"
    
    start_load = time.time()
    if not onnx_path.exists():
        logger.info("Exporting model to ONNX for CPU...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True, cache_dir=CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        model.save_pretrained(onnx_path)
        tokenizer.save_pretrained(onnx_path)
    
    # 명시적으로 CPU Provider 사용
    model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, provider="CPUExecutionProvider")
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    load_time = time.time() - start_load
    logger.info(f"Model Load Time: {load_time:.2f}s")
    
    def embed_onnx(texts):
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        model_output = model(**encoded_input)
        # Mean Pooling 구현
        token_embeddings = model_output[0]
        input_mask_expanded = np.expand_dims(encoded_input['attention_mask'], -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    # 워밍업
    embed_onnx(["Warmup text"])
    
    start_inference = time.time()
    for _ in range(5):
        # 100개를 한 번에 배치 처리
        _ = embed_onnx(TEST_TEXTS)
    inference_time = (time.time() - start_inference) / 5
    
    logger.info(f"Average Inference Time (100 docs): {inference_time:.4f}s")
    return load_time, inference_time

if __name__ == "__main__":
    standard_load, standard_inf = benchmark_standard_pytorch()
    onnx_load, onnx_inf = benchmark_optimum_onnx()
    
    if onnx_inf:
        print("\n" + "="*50)
        print(f"Standard PyTorch Inference: {standard_inf:.4f}s")
        print(f"Optimum ONNX Inference   : {onnx_inf:.4f}s")
        print(f"Speedup Factor           : {standard_inf/onnx_inf:.2f}x")
        print("="*50)
