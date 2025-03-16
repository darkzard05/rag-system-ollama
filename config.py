import os

# LLM 설정
LLM_MODEL = "gemma3:4b"
LLM_DEVICE = 'cuda'

# HuggingFaceEmbeddings 설정
EMBEDDINGS_MODEL_NAME = "intfloat/e5-base-v2"
EMBEDDINGS_MODEL_KWARGS = {'device': 'cpu'}

# 기타 설정
PDF_TEMP_SUFFIX = ".pdf"
PDF_LOAD_TIMEOUT = 600
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_SEARCH_KWARGS = {"k": 100}