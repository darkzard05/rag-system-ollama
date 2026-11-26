"""The grade prompt must describe the number of documents it actually receives.

``grade_top_n`` was widened from 3 to 5 to cover citations at rank 4-5, but
the prompt still told the grader it was looking at three documents.
"""

from __future__ import annotations

import inspect

from common.config import GRADE_PROMPT_CONFIG, GRADING_CONFIG
from core.graph import _grade_verify


def test_grade_prompt_label_matches_grade_top_n() -> None:
    top_n = int(GRADING_CONFIG.get("grade_top_n", 5))
    template = GRADE_PROMPT_CONFIG.get("human_message_template", "")
    assert f"[검색된 문서 (상위 {top_n}개)]" in template
    assert "[검색된 문서 (상위 3개)]" not in template


def test_code_fallback_label_matches_grade_top_n() -> None:
    source = inspect.getsource(_grade_verify.grade_documents)
    assert "[검색된 문서 (상위 3개)]" not in source
