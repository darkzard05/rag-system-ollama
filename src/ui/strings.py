"""Centralized UI string table with language switch support.

All user-facing strings are defined here instead of being scattered
across component modules.  Add a new language dict to ``UI_STRINGS``
to support additional locales.
"""

from __future__ import annotations

from common.config import MSG_CHAT_GUIDE, MSG_ERROR_OLLAMA_NOT_RUNNING

# ---------------------------------------------------------------------------
# Language switch
# ---------------------------------------------------------------------------

LANG: str = "en"


def set_lang(lang: str) -> None:
    """Switch the active UI language."""
    global LANG
    LANG = lang


# ---------------------------------------------------------------------------
# String tables
# ---------------------------------------------------------------------------

UI_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # Phase labels (chat.py _PROCESS_PHASES)
        "phase_document_search": "Document Search",
        "phase_evidence_collection": "Evidence Collection",
        "phase_answer_generation": "Answer Generation",
        # Streaming status
        "status_generating": "AI is generating answer... ▍",
        "status_stopped": "Generation stopped · No answer to display",
        "status_switching_models": "Switching models, please wait...",
        "status_generating_with_stop": (
            "AI is generating answer... · Stop with ■ button"
        ),
        # Chat input placeholder
        "chat_placeholder_followup": "Ask a follow-up question...",
        # Captions
        "caption_partial_answer": "Partial answer preserved",
        "caption_answer_complete": "Answer complete",
        "caption_stopped_partial": "Stopped · Partial answer preserved",
        # PDF viewer
        "pdf_error_data": "⚠️ Cannot load PDF data.",
        "pdf_error_open": (
            "⚠️ Cannot open PDF file. The file is corrupted or unsupported."
        ),
        "pdf_error_viewer": "PDF viewer error occurred. Please try again later.",
        "pdf_error_render": "PDF viewer rendering failed.",
        "pdf_highlight_load_failed": (
            "Cannot load highlights (coordinate cache read failed)."
        ),
        # Navigation
        "nav_prev": "◀ Previous",
        "nav_next": "Next ▶",
        # Chat guide
        "chat_guide": MSG_CHAT_GUIDE,
        # Ollama error
        "error_ollama_not_running": MSG_ERROR_OLLAMA_NOT_RUNNING,
        # Spinner
        "spinner_loading_models": "Loading available models…",
        # Expanders
        "expander_sources_reasoning": "Sources & Reasoning",
        "expander_sources_only": "Cited Sources",
        "expander_reasoning_only": "Thought Process",
    },
    "ko": {
        "phase_document_search": "문서 검색",
        "phase_evidence_collection": "증거 수집",
        "phase_answer_generation": "답변 생성",
        "status_generating": "AI가 답변을 생성 중입니다... ▍",
        "status_stopped": "답변 생성이 사용자에 의해 중단되었습니다.",
        "status_switching_models": "모델을 전환하는 중입니다...",
        "status_generating_with_stop": (
            "AI가 답변을 생성 중입니다... · ■ 버튼으로 중지할 수 있습니다"
        ),
        "chat_placeholder_followup": "이어서 질문을 입력하세요...",
        "caption_partial_answer": "작성된 내용까지 저장되었습니다",
        "caption_answer_complete": "답변 생성 완료",
        "caption_stopped_partial": "정지됨 · 부분적인 답변 보존됨",
        "pdf_error_data": "⚠️ PDF 데이터를 불러올 수 없습니다.",
        "pdf_error_open": (
            "⚠️ PDF 파일을 열 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식입니다."
        ),
        "pdf_error_viewer": "PDF 뷰어 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        "pdf_error_render": "PDF 뷰어 렌더링에 실패했습니다.",
        "pdf_highlight_load_failed": (
            "본문 위치 정보를 불러오지 못해 하이라이트 표시를 건너뜁니다."
        ),
        "nav_prev": "⬅️ 이전",
        "nav_next": "다음 ➡️",
        "chat_guide": MSG_CHAT_GUIDE,
        "error_ollama_not_running": MSG_ERROR_OLLAMA_NOT_RUNNING,
        "spinner_loading_models": "사용 가능한 모델 목록을 불러오는 중...",
        # Expanders
        "expander_sources_reasoning": "출처 및 추론 과정",
        "expander_sources_only": "인용 출처",
        "expander_reasoning_only": "사고 과정",
    },
}


def t(key: str) -> str:
    """Look up a UI string by key for the active language."""
    return UI_STRINGS.get(LANG, UI_STRINGS["en"]).get(key, key)


# ---------------------------------------------------------------------------
# Phase keywords (used for status-text matching, language-independent)
# ---------------------------------------------------------------------------

PHASE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "phase_document_search",
        ("검색", "fetch", "retriev", "문서"),
    ),
    (
        "phase_evidence_collection",
        ("증거", "연결", "context", "evidence", "리랭크", "rerank"),
    ),
    (
        "phase_answer_generation",
        ("답변", "생성", "generate", "response", "작성"),
    ),
)


def get_phase_labels() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return phase labels resolved for the active language."""
    return tuple((t(key), keywords) for key, keywords in PHASE_KEYWORDS)
