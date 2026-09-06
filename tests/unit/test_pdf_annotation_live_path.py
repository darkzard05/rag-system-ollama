"""A1 — PDF 주석 라이브 경로 검증.

C14 통합 후 ``consume_stream_into_message``는 완료 턴의 ``_finalize_pdf_
side_effects``를 동기 호출하므로(기존 백그라운드 스레드 finally 역할),
문서를 로드한 세션에서 스트림을 소비하면 ``pdf_annotations``가 올바른
``file_hash``와 함께 영속화된다. 문서가 실제 fitz 좌표를 가질 필요 없도록
``ui.components.streaming.extract_annotations_from_docs``를 패치해 결정적
주석을 반환하게 한다.
"""

from types import SimpleNamespace
from unittest.mock import patch

from core.session import SessionManager


def _fake_chunk(
    content: str = "",
    thought: str = "",
    status: str | None = None,
    metadata: dict | None = None,
    performance: dict | None = None,
) -> SimpleNamespace:
    """StreamChunk와 동일한 속성을 가진 경량 fake 청크."""
    return SimpleNamespace(
        status=status,
        thought=thought,
        content=content,
        metadata=metadata,
        performance=performance,
    )


def test_consume_stream_writes_pdf_annotations_with_file_hash():
    """Given PDF가 로드된 세션 / When 스트림을 소비 / Then pdf_annotations가
    올바른 file_hash와 함께 저장되고 메시지는 오류 없이 영속화된다."""
    sid = "test_pdf_annotation_live"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("pdf_file_path", "C:/docs/report.pdf", sid)
    SessionManager.set("file_hash", "abc123hash", sid)

    fake_docs = [
        {
            "metadata": {"page": 1, "source": "C:/docs/report.pdf"},
            "page_content": "hello",
        }
    ]

    def fake_stream(query: str, model_name: str, session_id: str):
        yield _fake_chunk(content="답변 내용", metadata={"documents": fake_docs})

    from ui.components.streaming import consume_stream_into_message

    deterministic_annotations = [{"type": "highlight", "page": 1}]
    with (
        patch(
            "ui.components.streaming.extract_annotations_from_docs",
            return_value=deterministic_annotations,
        ) as mock_extract,
        patch("ui.components.streaming.stream_chunks", fake_stream),
    ):
        result = consume_stream_into_message(sid, "질문", "test-model")

    # 메시지 영속화 + 오류 없음
    assert result is not None
    assert "error" not in result
    assert result["content"] == "답변 내용"
    assert result["documents"] == fake_docs

    # PDF 주석이 올바른 file_hash와 함께 저장됨
    annotations = SessionManager.get("pdf_annotations", None, sid)
    assert annotations is not None
    assert annotations["file_hash"] == "abc123hash"
    assert annotations["annotations"] == deterministic_annotations
    mock_extract.assert_called_once()
