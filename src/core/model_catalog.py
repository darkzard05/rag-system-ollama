"""Lightweight model catalog and filtering utility.

Keeps model filtering logic import-light (no numpy, no langchain, no heavy torch/ollama)
so that sidebar rendering can classify LLMs and embeddings without triggering
heavy model_loader initializations.
"""

from __future__ import annotations

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)


def get_filtered_models(available_models: list[str]) -> dict[str, list[str]]:
    """모델 목록을 LLM과 임베딩 모델로 분류하여 반환합니다."""
    safe_models = [m for m in available_models if m and "---" not in str(m)]
    embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

    embedding_candidates = [
        m for m in safe_models if any(kw in str(m).lower() for kw in embed_keywords)
    ]
    actual_embeddings = sorted(set(embedding_candidates))
    if DEFAULT_EMBEDDING_MODEL not in actual_embeddings:
        actual_embeddings.append(DEFAULT_EMBEDDING_MODEL)
    actual_embeddings.sort()

    llm_candidates = [m for m in safe_models if m not in embedding_candidates]
    # 중복 제거
    actual_llms = (
        sorted(set(llm_candidates)) if llm_candidates else [DEFAULT_OLLAMA_MODEL]
    )
    if DEFAULT_OLLAMA_MODEL not in actual_llms:
        actual_llms.append(DEFAULT_OLLAMA_MODEL)
    actual_llms.sort()

    return {"llm": actual_llms, "embedding": actual_embeddings}
