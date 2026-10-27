"""Shim: autoscroll_edge -> chat_autoscroll."""

from ui.components.chat_autoscroll import BOTTOM_STICK_PX, edge_latch

THRESHOLD = BOTTOM_STICK_PX

__all__ = ["THRESHOLD", "edge_latch"]
