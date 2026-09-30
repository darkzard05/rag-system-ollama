"""Stick-math autoscroll helper (pure Python, no Streamlit)."""

# Single source of truth, also injected into the JS in ``ui.py``. Deliberately
# small: a wide value creates a snap zone where scrolling down releases the
# latch and yanks the view the rest of the way to the bottom.
BOTTOM_STICK_PX: int = 8


def stick_math(
    scroll_height: int,
    scroll_top: int,
    client_height: int,
    threshold: int = BOTTOM_STICK_PX,
) -> dict[str, bool | int]:
    """Compute distance to bottom and whether stick-to-bottom is active."""
    if scroll_height <= 0 or client_height <= 0:
        return {"dist": -1, "stick_active": False}
    dist: int = scroll_height - scroll_top - client_height
    return {"dist": dist, "stick_active": dist <= threshold}
