"""Chonkie chunking adapter (Phase 3: v2-moe embedding integration).

Optional dependency (``pip install -r requirements/optional-chonkie.txt``).
Header-aware wrapping: split by markdown headers first, delegate each
section body to chonkie, then re-attach ``current_section``/``section_path``
metadata so downstream (retrieval/citations) keeps working unchanged.

Embedding: project Ollama ``nomic-embed-text-v2-moe`` wrapped in chonkie's
``BaseEmbeddings`` with L3 disk-cache passthrough (same
``emb:{model}:{xxh64}`` key scheme as ``core.semantic_chunker``).
``minishlab/potion-base-32M`` is the fallback when Ollama is unavailable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import xxhash
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)

# chonkie>=1.0.0 하한 (requirements/optional-chonkie.txt). 상한핀은 해당 파일 참조.
_CHONKIE_MIN_VERSION = (1, 0, 0)


def split_by_headers(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, body) pairs (header-aware)."""
    parts = _HEADER_RE.split(markdown)
    # parts: [pre, h1, body1, h2, body2, ...]
    if len(parts) == 1:
        return [("", markdown)]
    out: list[tuple[str, str]] = []
    pre = parts[0].strip()
    if pre:
        out.append(("", pre))
    it = iter(parts[1:])
    for header, body in zip(it, it, strict=False):
        title = header.lstrip("#").strip()
        out.append((title, body))
    return out


def is_chonkie_available() -> bool:
    """Return True when ``chonkie`` is importable."""
    try:
        import chonkie  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True


class OllamaV2MoeEmbeddings:
    """v2-moe Ollama embedder with L3 disk-cache passthrough.

    chonkie ``BaseEmbeddings`` protocol (``embed``/``embed_batch``/
    ``dimension``) against the project's ``load_embedding_model``.
    Cache misses are sent as a single Ollama request (Ollama sends the
    whole input in one HTTP round-trip — splitting only adds latency).
    """

    def __init__(self, model_name: str = "nomic-embed-text-v2-moe") -> None:
        from chonkie.embeddings import BaseEmbeddings  # type: ignore[import-not-found]

        # chonkie 1.7.0 checks isinstance against BaseEmbeddings internally;
        # register dynamically to avoid import-time hard dependency.
        if not isinstance(self, BaseEmbeddings):
            BaseEmbeddings.register(OllamaV2MoeEmbeddings)

        from common.config import MODEL_CACHE_DIR
        from core.model_loader import load_embedding_model
        from services.optimization import SyncCacheBridge
        from services.optimization.backends import DiskCache

        self.model_name = model_name
        self._embedder = load_embedding_model(model_name)
        cache_dir = str(Path(MODEL_CACHE_DIR) / "embedding_cache")
        # sync L1 (프로세스 내 dict) + L3 (semantic_chunker와 동일 디스크 경로).
        # DiskCache는 CacheBackend(async get/set)라 SyncCacheBridge로 감싼다.
        self._l1: dict[str, np.ndarray] = {}
        # NOTE: SyncCacheBridge는 실행 시 get/set 덕타이핑만 사용하므로
        # DiskCache도 동작한다 (어노테이션만 ObjectCache로 좁음).
        self._bridge = SyncCacheBridge(DiskCache(cache_dir=cache_dir))  # type: ignore[arg-type]
        self._dim: int | None = None

    def _key(self, norm_text: str) -> str:
        return f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"

    @staticmethod
    def _as_vector(raw: Any) -> np.ndarray | None:
        try:
            vec = raw["vector"] if isinstance(raw, dict) and "vector" in raw else raw
            arr = np.asarray(vec, dtype="float32")
        except (TypeError, ValueError):
            return None
        return arr if arr.ndim == 1 and arr.shape[0] else None

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch with L3 lookup + single-request miss fill."""
        norms = [" ".join(t.split()) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)
        missing: list[str] = []
        missing_idx: list[int] = []
        for i, norm in enumerate(norms):
            hit = self._l1.get(norm)
            if hit is None:
                try:
                    hit = self._as_vector(self._bridge.get_sync(self._key(norm)))
                except Exception as e:
                    logger.debug(f"[chonkie] L3 lookup skipped: {e}")
                    hit = None
            if hit is not None:
                self._l1[norm] = hit
                results[i] = hit
            else:
                missing.append(norm)
                missing_idx.append(i)
        if missing:
            vecs = self._embedder.embed_documents(missing)
            for norm, vec, i in zip(missing, vecs, missing_idx, strict=True):
                arr = np.asarray(vec, dtype="float32")
                results[i] = arr
                self._l1[norm] = arr
                try:
                    self._bridge.set_sync(
                        self._key(norm),
                        {"vector": arr.tolist(), "cache_version": "1.0"},
                    )
                except Exception as e:
                    logger.debug(f"[chonkie] L3 cache store skipped: {e}")
        return [r for r in results if r is not None]

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text (batch path, cache-shared)."""
        out = self.embed_batch([text])
        if not out:
            raise ValueError("[chonkie] v2-moe embedding returned empty result")
        return out[0]

    async def aembed(self, text: str) -> np.ndarray:
        """Async single embed (thread-offloaded, loop-safe)."""
        import asyncio

        return await asyncio.to_thread(self.embed, text)

    async def aembed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Async batch embed (thread-offloaded, loop-safe)."""
        import asyncio

        return await asyncio.to_thread(self.embed_batch, texts)

    @staticmethod
    def similarity(u: np.ndarray, v: np.ndarray) -> np.float32:
        """Cosine similarity (mirrors chonkie BaseEmbeddings default)."""
        return np.float32(
            np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v)),
        )

    def get_tokenizer(self) -> Any:
        """chonkie chunk_size 측정에 쓰는 v2-moe 토크나이저 (실패 시 문자 단위)."""
        try:
            from chonkie.tokenizer import (
                AutoTokenizer,  # type: ignore[import-not-found]
            )

            return AutoTokenizer("nomic-ai/nomic-embed-text-v2-moe")
        except Exception as e:
            logger.warning(f"[chonkie] HF tokenizer 실패, 문자 단위 폴백: {e}")
            from chonkie.tokenizer import (
                CharacterTokenizer,  # type: ignore[import-not-found]
            )

            return CharacterTokenizer()

    @property
    def dimension(self) -> int:
        """Embedding dimension (lazy probe, v2-moe=768)."""
        if self._dim is None:
            try:
                self._dim = int(self.embed("dimension probe").shape[0])
            except Exception:
                self._dim = 768
        return self._dim


def _build_embedder() -> Any:
    """Prefer v2-moe Ollama wrapper, fall back to potion-base-32M."""
    try:
        emb = OllamaV2MoeEmbeddings()
        logger.info("[chonkie] embedding: v2-moe Ollama (L3 cached)")
        return emb
    except Exception as e:
        logger.warning(f"[chonkie] v2-moe wrapper failed, potion fallback: {e}")
        return "minishlab/potion-base-32M"


def chunk_with_chonkie(
    docs: list[Document],
    chunk_size: int = 512,
    threshold: float = 0.7,
    similarity_window: int = 3,
    min_sentences_per_chunk: int = 2,
    skip_window: int = 0,
    filter_window: int = 5,
    filter_polyorder: int = 3,
    filter_tolerance: float = 0.2,
    overlap_ratio: float = 0.1,
) -> list[Document]:
    """Chunk documents with chonkie SemanticChunker (header-aware).

    Defaults are the Phase4 sweep center (threshold 0.7 / window 3 /
    skip 0). Raises ImportError when chonkie is not installed.
    """
    try:
        from chonkie import (
            OverlapRefinery,  # type: ignore[import-not-found]
            SemanticChunker,  # type: ignore[import-not-found]
        )
    except ImportError as e:
        raise ImportError(
            "chonkie backend requested but 'chonkie' is not installed. "
            "Install with: pip install -r requirements/optional-chonkie.txt"
        ) from e

    chunker = SemanticChunker(
        embedding_model=_build_embedder(),
        threshold=threshold,
        chunk_size=chunk_size,
        similarity_window=similarity_window,
        min_sentences_per_chunk=min_sentences_per_chunk,
        skip_window=skip_window,
        filter_window=filter_window,
        filter_polyorder=filter_polyorder,
        filter_tolerance=filter_tolerance,
    )
    try:
        refinery = OverlapRefinery(context_size=overlap_ratio)
    except Exception as e:
        logger.warning(f"[chonkie] refinery disabled: {e}")
        refinery = None
    out: list[Document] = []
    for doc in docs:
        base_meta = dict(doc.metadata or {})
        for section, body in split_by_headers(doc.page_content):
            if not body.strip():
                continue
            try:
                chunks = chunker.chunk(body)
                if refinery is not None and chunks:
                    try:
                        chunks = refinery.refine(chunks)
                    except Exception as e:
                        logger.debug(f"[chonkie] refinery skipped: {e}")
            except Exception as e:
                logger.warning(f"[chonkie] section chunk failed, keep raw: {e}")
                chunks = []
            if not chunks:
                meta = {
                    **base_meta,
                    "current_section": section or base_meta.get("current_section", ""),
                }
                meta["section_path"] = meta["current_section"]
                out.append(Document(page_content=body.strip(), metadata=meta))
                continue
            for ch in chunks:
                text = getattr(ch, "text", str(ch)).strip()
                if not text:
                    continue
                current = section or base_meta.get("current_section", "")
                meta = {
                    **base_meta,
                    "current_section": current,
                    "section_path": current,
                    "chunk_method": "chonkie-semantic",
                    "token_count": getattr(ch, "token_count", None),
                }
                out.append(Document(page_content=text, metadata=meta))
    return out
