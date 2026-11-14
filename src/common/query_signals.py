"""Query signal helpers shared by preprocess and generate.

``preprocess`` uses these signals to pick ensemble weights, while ``generate``
uses the keyword signal to skip the short-answer regeneration pass for bare
term lookups (e.g. ``cm3``). Keeping the classification in one module prevents
the two call sites from drifting apart.
"""

from __future__ import annotations

import re

from common.config import DYNAMIC_WEIGHTING_CONFIG

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _is_bare_identifier(query: str) -> bool:
    """Return True for a single, whitespace-free technical identifier token.

    The configured keyword patterns only cover digits/versions/hex, so bare
    hyphenated terms such as ``vqvae-gan`` or ``vq-vae`` slip through. Those
    are still term lookups whose short answers are normal and should not
    trigger an expensive expansion regeneration. Multi-hyphen phrases like
    ``state-of-the-art`` are excluded to avoid treating ordinary compound
    words as term lookups.
    """
    q = query.strip()
    if not q or any(ch.isspace() for ch in q):
        return False
    if not 3 <= len(q) <= 24:
        return False
    if not _IDENTIFIER_RE.fullmatch(q):
        return False
    if any(ch.isdigit() for ch in q):
        return True
    if q.count("-") != 1:
        return False
    left, right = q.split("-")
    return len(left) >= 2 and len(right) >= 2


def is_keyword_heavy_query(query: str) -> bool:
    """Return True when the query is dominated by keyword/identifier patterns."""
    if not DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):
        return False
    patterns = DYNAMIC_WEIGHTING_CONFIG.get("keyword_patterns", [])
    return any(re.search(pattern, query) for pattern in patterns)


def is_semantic_heavy_query(query: str) -> bool:
    """Return True when the query contains an explicit semantic-intent keyword."""
    if not DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):
        return False
    keywords = DYNAMIC_WEIGHTING_CONFIG.get("semantic_keywords", [])
    return any(keyword in query for keyword in keywords)


def is_term_lookup_query(query: str) -> bool:
    """Keyword-heavy or bare identifier, but not semantic-heavy.

    Examples: ``cm3``, ``resnet50`` (keyword patterns) and ``vqvae-gan``,
    ``vq-vae`` (bare hyphenated identifiers) are term lookups. Semantic
    questions such as ``vqvae-gan이 뭔가요`` are not.
    """
    if is_semantic_heavy_query(query):
        return False
    return is_keyword_heavy_query(query) or _is_bare_identifier(query)
