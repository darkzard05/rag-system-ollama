"""Query signal helpers shared by preprocess and generate.

``preprocess`` uses these signals to pick ensemble weights, while ``generate``
uses the keyword signal to skip the short-answer regeneration pass for bare
term lookups (e.g. ``cm3``). Keeping the classification in one module prevents
the two call sites from drifting apart.
"""

from __future__ import annotations

import re

from common.config import DYNAMIC_WEIGHTING_CONFIG


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
    """Keyword-heavy but not semantic-heavy — a bare term lookup like ``cm3``."""
    return is_keyword_heavy_query(query) and not is_semantic_heavy_query(query)
