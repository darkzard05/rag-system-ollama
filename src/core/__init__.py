"""Core package: keep this module import-light.

Eagerly importing ``core.graph.graph_builder`` here pulls in LangGraph and
langchain_ollama on the first ``core.*`` import, delaying the Streamlit
first paint.
"""

from typing import Any

__all__ = ["graph_builder"]


def __getattr__(name: str) -> Any:
    if name == "graph_builder":
        from core.graph import graph_builder as _graph_builder

        return _graph_builder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
