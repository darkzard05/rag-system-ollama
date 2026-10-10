"""Document analysis panel (greenfield, 1A+2B+3B+4A).

Public API is a single entry point: ``render_document_panel(sid)``.
``BuildState`` is the sole source of truth — flat legacy keys
(``rebuild_progress``/``rebuild_status``/``rebuild_done``) are never read here.
"""

from ui.components.doc_analysis.panel import render_document_panel
from ui.components.doc_analysis.viewmodel import BuildVM, to_viewmodel

__all__ = ["BuildVM", "render_document_panel", "to_viewmodel"]
