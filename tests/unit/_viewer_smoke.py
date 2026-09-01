"""Headless viewer render script for unit tests (no full app boot).

Driven by ``AppTest.from_file``. Renders only the PDF viewer fragment in an
empty session (no PDF loaded) to lock its deterministic top-level structure
(C8 guard: 최상위 요소 수/순서 불변) and to verify it renders without a
``run_every`` polling timer (fix-001-polling-fragments).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

os.environ.setdefault("IS_CI_TEST", "true")

from ui.components.viewer import render_pdf_area  # noqa: E402

render_pdf_area()
