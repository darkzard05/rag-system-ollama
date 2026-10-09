"""Centralized UI string table with language switch support.

All user-facing strings are defined here instead of being scattered
across component modules.  Add a new language dict to ``UI_STRINGS``
to support additional locales.
"""

from __future__ import annotations

import ast
from pathlib import Path

from common.config import MSG_ERROR_OLLAMA_NOT_RUNNING

# ---------------------------------------------------------------------------
# Language switch
# ---------------------------------------------------------------------------

LANG: str = "en"


def set_lang(lang: str) -> None:
    """Switch the active UI language."""
    global LANG
    LANG = lang


# ---------------------------------------------------------------------------
# Dropped-key tracking (WARNING counter — no silent OSError/UnicodeDecodeError drop)
# ---------------------------------------------------------------------------

_dropped_key_counter: int = 0


def dropped_key_count() -> int:
    """Return the number of times ``t(key`` was called with a missing key."""
    return _dropped_key_counter


def _record_dropped_key() -> None:
    """Increment the dropped-key WARNING counter."""
    global _dropped_key_counter
    _dropped_key_counter += 1


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
        "status_stopped": "Generation stopped · No answer to display",
        # Captions
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
        # PDF page input (viewer.py render_pdf_controls, visible label + help)
        "pdf_page_label": "Page",
        "pdf_page_help": "Page number to jump to (1–N).",
        # Onboarding guide (chat_build._render_guidance_panel, single source)
        "onboarding_guide": (
            "Answers with cited evidence from your uploaded PDF.\n\n"
            "**How to start:**\n"
            "1. **Upload** a PDF document from the left sidebar.\n"
            "2. Once indexing finishes, freely ask anything about the document.\n"
            "3. Press a **citation button** in an AI answer to jump to that page."
        ),
        # Sidebar (sidebar.py, LANG toggle host — en values mirror the
        # pre-i18n literals so the default LANG="en" render is unchanged)
        "sidebar_upload_label": "Upload PDF Document",
        "sidebar_settings": "Settings",
        "sidebar_llm_label": "LLM Model Selection",
        "sidebar_ollama_hint": (
            "Start Ollama to select a model. "
            "[Installation guide](https://ollama.com/download)"
        ),
        "sidebar_refresh": "Refresh Models",
        "sidebar_refresh_help": "Refresh the list of models available in Ollama.",
        "sidebar_embedding_label": "Embedding Model Selection",
        "sidebar_new_chat": "New Chat",
        "sidebar_new_chat_help": "Start a new chat, keeping the uploaded documents.",
        "sidebar_reset": "🗑️ Reset All",
        "sidebar_reset_help": "Delete all conversations and data.",
        "sidebar_reset_title": "Confirm Reset All",
        "sidebar_reset_body": (
            "This will delete all conversation history and uploaded document "
            "data. Do you want to continue?"
        ),
        "sidebar_cancel": "Cancel",
        "sidebar_confirm_reset": "Reset",
        "sidebar_switching": "⏳ Switching model... Please wait.",
        "sidebar_language": "Language / 언어",
        "sidebar_language_help": "Switch the UI language (UI 언어 전환).",
        "nav_next": "Next ▶",
        # Ollama error
        "error_ollama_not_running": MSG_ERROR_OLLAMA_NOT_RUNNING,
        # Spinner
        "spinner_loading_models": "Loading available models…",
        # Expanders
        "expander_sources_reasoning": "Sources & Reasoning",
        "expander_sources_only": "Cited Sources",
        # Retry CTA (stopped-answer dead-end, mirrors build-error CTA shape)
        "action_retry_answer": "Retry answer",
        "action_retry_answer_help": "Ask the last question again",
        # Retry CTA (build-error dead-end, mirrors stopped-answer CTA shape)
        "action_retry_build": "Retry Analysis",
        "action_retry_build_help": "Rebuild the index from the uploaded PDF",
        "action_cancel_build": "Cancel Analysis",
        "expander_reasoning_only": "Thought Process",
        # Reference ghost skeleton (chat_references._render_references_content)
        "ref_ghost_searching": "🔍 searching p.{page} ▍",
        "ref_jump_help": "Go to PDF page {page}",
        "ref_section_title": "Reference",
        # Streaming status fallbacks (chat._draw_streaming_message, streaming_core)
        "status_initial_search": "Analyzing the question and searching related knowledge... ▍",
        "status_fallback_generating": "Generating...",
        "status_fallback_stopping": "Stopping...",
        "status_citation_cleanup": "Tidying citations...",
        "error_prefix": "Error: ",
        # Document build progress (chat_build._render_build_progress_block)
        "doc_default_name": "Document",
        "cache_tag_reused": " (reusing previous analysis)",
        "cache_tag_new": " [new]",
        "build_cancelled": "Cancelling analysis...",
        "build_analyzing": "Building knowledge base...",
        "build_in_progress": "Analyzing document",
        "build_progress_caption": "%{progress}% complete · {elapsed}s elapsed",
        "build_complete_caption": "Analysis took {elapsed}s",
        "build_complete_default": "Analysis complete",
        # Chat input placeholders (chat._resolve_chat_input_state)
        "chat_placeholder_generating": "Generating an answer... (■ button on the right of the input to stop)",
        "chat_placeholder_swapping": "Switching the AI model... Please wait a moment.",
        "chat_placeholder_building": "Analyzing document knowledge... (you can ask once done)",
        "chat_placeholder_no_pdf": "Please upload a PDF document from the left sidebar first.",
        "chat_placeholder_not_ready": "Preparing the document index. Please wait a moment.",
        "chat_placeholder_ready": "Ask anything about the document.",
        "chat_placeholder_next": "Type a follow-up question...",
    },
    "ko": {
        "phase_document_search": "문서 검색",
        "phase_evidence_collection": "증거 수집",
        "phase_answer_generation": "답변 생성",
        "status_stopped": "답변 생성이 사용자에 의해 중단되었습니다.",
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
        "pdf_page_label": "페이지",
        "pdf_page_help": "이동할 페이지 번호 (1–N).",
        "onboarding_guide": (
            "업로드하신 PDF 문서의 내용을 바탕으로 정확한 근거와 함께 답변해 드립니다.\n\n"
            "**시작하는 방법:**\n"
            "1. **좌측 사이드바**에서 분석할 PDF 문서를 업로드해 주세요.\n"
            "2. 지식 베이스 구축이 완료되면 본문에 대한 질문을 자유롭게 입력하세요.\n"
            "3. AI 답변과 함께 제공되는 **인용 출처 버튼**을 누르면 해당 페이지로 즉시 이동합니다."
        ),
        "sidebar_upload_label": "PDF 문서 업로드",
        "sidebar_settings": "설정",
        "sidebar_llm_label": "LLM 모델 선택",
        "sidebar_ollama_hint": (
            "모델을 선택하려면 Ollama를 시작하세요. "
            "[설치 안내](https://ollama.com/download)"
        ),
        "sidebar_refresh": "모델 새로고침",
        "sidebar_refresh_help": "Ollama에서 사용 가능한 모델 목록을 새로고침합니다.",
        "sidebar_embedding_label": "임베딩 모델 선택",
        "sidebar_new_chat": "새 대화",
        "sidebar_new_chat_help": "업로드된 문서는 유지하고 새 대화를 시작합니다.",
        "sidebar_reset": "🗑️ 전체 초기화",
        "sidebar_reset_help": "모든 대화와 데이터를 삭제합니다.",
        "sidebar_reset_title": "전체 초기화 확인",
        "sidebar_reset_body": (
            "현재까지의 모든 대화 기록과 업로드된 문서 데이터가 삭제됩니다. "
            "계속하시겠습니까?"
        ),
        "sidebar_cancel": "취소",
        "sidebar_confirm_reset": "초기화 실행",
        "sidebar_switching": "⏳ 모델 전환 중입니다. 잠시만 기다려 주세요.",
        "sidebar_language": "Language / 언어",
        "sidebar_language_help": "UI 언어를 전환합니다 (Switch the UI language).",
        "nav_next": "다음 ➡️",
        "error_ollama_not_running": MSG_ERROR_OLLAMA_NOT_RUNNING,
        "spinner_loading_models": "사용 가능한 모델 목록을 불러오는 중...",
        # Expanders
        "expander_sources_reasoning": "출처 및 추론 과정",
        "expander_sources_only": "인용 출처",
        "action_retry_answer": "다시 답변 생성",
        "action_retry_answer_help": "마지막 질문을 다시 요청합니다",
        "action_retry_build": "다시 시도",
        "action_retry_build_help": "업로드한 PDF로 인덱스를 다시 만듭니다",
        "action_cancel_build": "분석 취소",
        "expander_reasoning_only": "사고 과정",
        # Reference ghost skeleton (chat_references._render_references_content)
        "ref_ghost_searching": "🔍 p.{page} 검색 중 ▍",
        "ref_jump_help": "PDF {page}페이지로 이동",
        "ref_section_title": "Reference",
        # Streaming status fallbacks (chat._draw_streaming_message, streaming_core)
        "status_initial_search": "질문을 분석하고 관련 지식을 검색하는 중입니다... ▍",
        "status_fallback_generating": "생성 중...",
        "status_fallback_stopping": "중지 중...",
        "status_citation_cleanup": "인용 정리 중...",
        "error_prefix": "오류: ",
        # Document build progress (chat_build._render_build_progress_block)
        "doc_default_name": "문서",
        "cache_tag_reused": " (기존 분석 재사용)",
        "cache_tag_new": " [new]",
        "build_cancelled": "분석 취소 중...",
        "build_analyzing": "지식 베이스 구축 중...",
        "build_in_progress": "문서 분석 중",
        "build_progress_caption": "{progress}% 완료 · {elapsed}초 경과",
        "build_complete_caption": "분석 소요 시간: {elapsed}초",
        "build_complete_default": "분석 완료",
        # Chat input placeholders (chat._resolve_chat_input_state)
        "chat_placeholder_generating": "답변을 생성하고 있습니다... (중지는 입력창 우측 ■ 버튼)",
        "chat_placeholder_swapping": "AI 모델을 전환하는 중입니다... 잠시만 기다려 주세요.",
        "chat_placeholder_building": "문서 지식을 분석하고 있습니다... (완료 후 질문 가능)",
        "chat_placeholder_no_pdf": "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요.",
        "chat_placeholder_not_ready": "문서 인덱스를 준비 중입니다. 잠시만 기다려 주세요.",
        "chat_placeholder_ready": "문서 내용에 대해 궁금한 점을 질문해 보세요.",
        "chat_placeholder_next": "이어서 추가 질문을 입력하세요...",
    },
}


def t(key: str) -> str:
    """Look up a UI string by key for the active language.

    If the key is not found in the active language table, a WARNING is
    recorded via ``_record_dropped_key()`` (no silent OSError/UnicodeDecodeError
    drop — the key is still returned as fallback so rendering is not broken,
    but the caller can check ``dropped_key_count()``).
    """
    result = UI_STRINGS.get(LANG, UI_STRINGS["en"]).get(key, None)
    if result is None:
        _record_dropped_key()
        return key
    return result


# ---------------------------------------------------------------------------
# AST‑based key collection (both‑direction i18n gate)
# ---------------------------------------------------------------------------

_DirectKeySet = set[str]


def _collect_t_keys_ast(root: Path) -> _DirectKeySet:
    """Scan Python source under *root* for literal ``t("key")`` calls.

    Only ``ast.parse`` literal calls are counted — comments and string
    concatenation are excluded (comment‑spoof proof).
    """
    found: _DirectKeySet = set()
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "t"
            ):
                # Collect only literal string arguments; skip non-literal args
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        found.add(arg.value)
    return found


def _indirect_keys() -> set[str]:
    """Keys consumed via ``PHASE_KEYWORDS`` (resolved dynamically in ``t``)."""
    return {key for key, _keywords in PHASE_KEYWORDS}


def gate_en_keys_read() -> tuple[set[str], set[str]]:
    """i18n gate: every ``en`` key must be read directly or via ``PHASE_KEYWORDS``.

    Returns
        missing: keys in ``UI_STRINGS["en"]`` not found in any ``t()`` call or
                 ``PHASE_KEYWORDS``
        absent_phase: ``PHASE_KEYWORDS`` keys missing from ``UI_STRINGS["en"]``
    """
    root = Path(__file__).resolve().parents[2] / "src"
    direct_found = _collect_t_keys_ast(root)
    indirect = _indirect_keys()
    readable = direct_found | indirect
    missing = set(UI_STRINGS["en"]) - readable
    absent_phase = indirect - set(UI_STRINGS["en"])
    return missing, absent_phase


def gate_tables_symmetric() -> bool:
    """i18n gate: the ``en`` and ``ko`` tables must carry the same key set."""
    return set(UI_STRINGS["en"]) == set(UI_STRINGS["ko"])


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
