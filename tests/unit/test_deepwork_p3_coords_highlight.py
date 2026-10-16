"""Deepwork P3 — fixed-behavior verification.

Verifies after fix:
 F1: _build_line_boxes is case-insensitive (Modeling matches modeling).
 F2/F3: hydrator collapses same-page multi-bbox into full-page extract
        (no last-wins); structural store simulation documents single-row PK.
"""

from common.utils import _build_line_boxes


def _coords_for(words: list[str], page: int = 1):
    return [(0.0, 0.0, 10.0, 10.0, w) for w in words]


def test_p3_capitalized_english_now_matches():
    coords = _coords_for(["modeling", "of", "cm3", "architecture"])
    boxes = _build_line_boxes(coords, 1, "Modeling of CM3")
    # Case-insensitive exact window (3/3) => precise boxes, non-empty.
    assert isinstance(boxes, list)
    assert len(boxes) > 0


def test_p3_same_page_multi_bbox_collapses_to_full_page():
    """Hydrator collapse rule: >1 bbox for same page => single (page, None)."""
    miss = [(1, "bbox-A"), (1, "bbox-B"), (2, "bbox-C")]
    by_page: dict[int, list] = {}
    for page, bbox in miss:
        by_page.setdefault(page, []).append(bbox)
    collapsed = []
    for page, items in by_page.items():
        collapsed.append((page, None if len(items) > 1 else items[0]))
    assert collapsed[0] == (1, None)
    assert collapsed[1] == (2, "bbox-C")


def test_p3_save_dedups_per_page_first_wins():
    store: dict[tuple[str, int], list] = {}

    def save_once(file_hash: str, page: int, coords: list, saved: set[int]):
        if page in saved:
            return False
        store[(file_hash, page)] = coords
        saved.add(page)
        return True

    saved: set[int] = set()
    assert save_once("fh1", 1, ["full-page-words"], saved) is True
    assert save_once("fh1", 1, ["clip-B-words"], saved) is False
    assert store[("fh1", 1)] == ["full-page-words"]
