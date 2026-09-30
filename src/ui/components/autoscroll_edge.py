"""Edge-latch autoscroll helper (pure Python, no Streamlit)."""

from ui.components.autoscroll_stick import BOTTOM_STICK_PX

THRESHOLD: int = BOTTOM_STICK_PX


def edge_latch(
    user_scrolled_up: bool, scroll_delta_y: int, dist_to_bottom: int
) -> bool:
    """Return True when the user scroll latch stays engaged."""
    return user_scrolled_up and (dist_to_bottom > THRESHOLD or scroll_delta_y != 0)
