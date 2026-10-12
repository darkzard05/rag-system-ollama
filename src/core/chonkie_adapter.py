"""Chonkie chunking adapter (spike, Phase 0-3).

Optional dependency (``pip install -r requirements/optional-chonkie.txt``).
Header-aware wrapping: split by markdown headers first, delegate each
section body to chonkie, then re-attach ``current_section`` metadata so
downstream (retrieval/citations) keeps working unchanged.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


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


def chunk_with_chonkie(
    docs: list[Document],
    chunk_size: int = 512,
    threshold: float = 0.5,
) -> list[Document]:
    """Chunk documents with chonkie SemanticChunker (header-aware).

    Raises ImportError when chonkie is not installed.
    """
    try:
        from chonkie import SemanticChunker  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "chonkie backend requested but 'chonkie' is not installed. "
            "Install with: pip install -r requirements/optional-chonkie.txt"
        ) from e

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",
        threshold=threshold,
        chunk_size=chunk_size,
    )
    out: list[Document] = []
    for doc in docs:
        base_meta = dict(doc.metadata or {})
        for section, body in split_by_headers(doc.page_content):
            if not body.strip():
                continue
            try:
                chunks = chunker.chunk(body)
            except Exception as e:
                logger.warning(f"[chonkie] section chunk failed, keep raw: {e}")
                chunks = []
            if not chunks:
                meta = {
                    **base_meta,
                    "current_section": section or base_meta.get("current_section", ""),
                }
                out.append(Document(page_content=body.strip(), metadata=meta))
                continue
            for ch in chunks:
                text = getattr(ch, "text", str(ch)).strip()
                if not text:
                    continue
                meta = {
                    **base_meta,
                    "current_section": section or base_meta.get("current_section", ""),
                    "chunk_method": "chonkie-semantic",
                    "token_count": getattr(ch, "token_count", None),
                }
                out.append(Document(page_content=text, metadata=meta))
    return out
