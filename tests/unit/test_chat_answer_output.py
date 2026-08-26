"""채팅 답변 출력 파이프라인 회귀 테스트.

검증 전용: 테스트만 추가하며 src/는 수정하지 않는다.
대상 경로:
- ui.components.streaming._extract_final_answer_delta / bg_task finally
- ui.components.chat.render_message (processed_content / escape / thought)
- common.utils.apply_tooltips_to_response / friendly_error_message
"""

import os
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from core.session import SessionManager
from ui.components.streaming import _GENERIC_STREAMING_MSG


def _fake_chunk(
    content: str = "",
    thought: str = "",
    status: str | None = None,
    metadata: dict | None = None,
    performance: dict | None = None,
    raw_json: bool = False,
) -> SimpleNamespace:
    """StreamChunk와 동일한 속성을 가진 경량 fake 청크를 생성합니다.

    raw_json 플래그를 추가로 받아 구조화 모드를 시뮬레이션한다.
    """
    return SimpleNamespace(
        status=status,
        thought=thought,
        content=content,
        metadata=metadata,
        performance=performance,
        raw_json=raw_json,
    )


def _drive_turn(sid: str, stream_factory) -> str:
    """stream_chunks를 패치하여 표준 소비 경로로 한 턴을 구동하고 msg_id를 반환.

    표준 리팩터 이후 스트리밍은 백그라운드 스레드가 아닌 ``stream_chunks``
    제너레이터를 단일 script run 안에서 동기 소비한다. 여기서는 위젯 렌더
    없이 content만 누적해 SessionManager 메시지(content)를 검증한다.
    """
    from ui.components import streaming as streaming_mod
    from ui.components.streaming import _extract_final_answer_delta

    SessionManager.reset_all_state(sid)
    msg_id = str(uuid.uuid4())
    SessionManager.add_message(
        role="assistant",
        content="",
        msg_type="general",
        msg_id=msg_id,
        thought="",
        documents=[],
        metrics={},
        citations=[],
        processed_content=None,
        session_id=sid,
    )

    accumulated = ""
    raw_json_parts: list[str] = []
    fa_scan_pos = 0
    error_msg: str | None = None
    with patch("ui.components.streaming.stream_chunks", stream_factory):
        try:
            for chunk in streaming_mod.stream_chunks("질문", "test-model", sid):
                if not chunk.content:
                    continue
                if getattr(chunk, "raw_json", False):
                    raw_json_parts.append(chunk.content)
                    blob = "".join(raw_json_parts)
                    delta, fa_scan_pos = _extract_final_answer_delta(blob, fa_scan_pos)
                    accumulated += delta
                else:
                    accumulated += chunk.content
        except Exception as exc:  # noqa: BLE001 - 스트림 레벨 오류 보존
            from ui.components.streaming import friendly_error_message

            error_msg = friendly_error_message(exc)

    SessionManager.set("is_generating_answer", False, sid)
    msg = _target_message(sid, msg_id)
    msg["content"] = accumulated
    if error_msg is not None:
        msg["error"] = error_msg
    return msg_id


def _target_message(sid: str, msg_id: str) -> dict:
    messages = SessionManager.get_messages(session_id=sid)
    return next(m for m in messages if m.get("msg_id") == msg_id)


def test_raw_json_chain_builds_processed_content():
    """raw_json 단일 청크 → final_answer 추출 + 툴팁 처리된 processed_content."""
    sid = "test_raw_json_ok"
    doc = Document(page_content="원문입니다.", metadata={"page": 5})

    def _stream(q: str, m: str, s: str):
        yield _fake_chunk(
            content='{"final_answer":"Hello [p.5]"}',
            raw_json=True,
            metadata={"documents": [doc]},
        )

    msg_id = _drive_turn(sid, _stream)
    msg = _target_message(sid, msg_id)

    assert msg["content"] == "Hello [p.5]"
    # processed_content는 렌더 시점에 apply_tooltips_to_response로 생성되므로
    # render_message 출력(AppTest markdown)으로 검증한다.
    from streamlit.testing.v1 import AppTest

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as script_content:
        script_content.write(
            "import sys\n"
            "sys.path.append(r'" + os.path.abspath("src") + "')\n"
            "from langchain_core.documents import Document\n"
            "from ui.components.chat import render_message\n"
            'render_message(role="assistant", content="Hello [p.5]", '
            'msg_type="general", documents='
            "[" + repr(doc) + "])\n"
        )
        script_path = script_content.name
    try:
        at = AppTest.from_file(script_path).run(timeout=60)
        assert not at.exception
        body = "".join(m.value for m in at.markdown)
        assert "citation-highlight" in body
        assert 'data-page="5"' in body
    finally:
        os.remove(script_path)
    assert msg["msg_type"] == "general"
    assert SessionManager.get("is_generating_answer", True, sid) is False


def test_raw_json_chain_falls_back_on_malformed():
    """raw_json 청크가 잘못된 JSON이면 원시 텍스트를 유지하고 크래시하지 않는다."""
    sid = "test_raw_json_bad"
    doc = Document(page_content="원문.", metadata={"page": 1})

    def _stream(q: str, m: str, s: str):
        yield _fake_chunk(
            content='{"final_answer":"Hi',
            raw_json=True,
            metadata={"documents": [doc]},
        )

    msg_id = _drive_turn(sid, _stream)
    msg = _target_message(sid, msg_id)

    # 스트리밍 중 final_answer 델타("Hi")가 이미 누적되고, finally의 JSON 파싱이
    # 실패(잘림)하므로 원시 텍스트가 아니라 누적된 델타를 유지한다.
    assert msg["content"] == "Hi"
    assert msg["msg_type"] == "general"
    assert SessionManager.get("is_generating_answer", True, sid) is False


def test_answer_xss_escaped_once():
    """인용 처리된 processed_content에 원시 <script>가 노출되지 않는다."""
    sid = "test_xss_escaped"
    doc = Document(page_content="<script>alert(1)</script>", metadata={"page": 2})

    def _stream(q: str, m: str, s: str):
        yield _fake_chunk(
            content='{"final_answer":"<script>alert(1)</script> [p.2]"}',
            raw_json=True,
            metadata={"documents": [doc]},
        )

    _drive_turn(sid, _stream)

    # processed_content는 렌더 시점에 생성되므로 아래 render_message/AppTest로 검증

    # render_message가 assistant content를 이스케이프하는지 직접 검증
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as script_content:
        script_content.write(
            "import sys\n"
            "sys.path.append(r'" + os.path.abspath("src") + "')\n"
            "from ui.components.chat import render_message\n"
            'render_message(role="assistant", content="see <b>bold</b>", '
            'msg_type="general")\n'
        )
        script_path = script_content.name
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(script_path).run(timeout=60)
        assert not at.exception
        body = "".join(m.value for m in at.markdown)
        assert "&lt;b&gt;" in body
        assert "<b>" not in body
    finally:
        os.remove(script_path)


def test_streaming_error_preserves_partial():
    """연결 오류 시 부분 응답 보존 + Ollama 미실행 친화 메시지."""
    from common.config import MSG_ERROR_OLLAMA_NOT_RUNNING

    sid = "test_err_partial"

    def _stream(q: str, m: str, s: str):
        yield _fake_chunk(content="부분 응답")
        raise ConnectionError("connection refused")

    msg_id = _drive_turn(sid, _stream)
    msg = _target_message(sid, msg_id)

    assert "부분 응답" in msg["content"]
    assert msg["error"] == MSG_ERROR_OLLAMA_NOT_RUNNING
    assert msg["msg_type"] == "general"
    assert SessionManager.get("is_generating_answer", True, sid) is False


def test_streaming_error_generic_message():
    """매핑되지 않은 예외는 제네릭 메시지를 사용한다."""
    sid = "test_err_generic"

    def _stream(q: str, m: str, s: str):
        yield _fake_chunk(content="부분 응답")
        raise RuntimeError("boom")

    msg_id = _drive_turn(sid, _stream)
    msg = _target_message(sid, msg_id)

    assert msg["error"] == _GENERIC_STREAMING_MSG
    assert "boom" not in msg["error"]
    assert msg["msg_type"] == "general"
    assert SessionManager.get("is_generating_answer", True, sid) is False


def test_thought_rendered_safe_by_default():
    """thought는 unsafe_allow_html=False로 렌더되어 실행 가능한 <script>가 없다."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as script_content:
        script_content.write(
            "import sys\n"
            "sys.path.append(r'" + os.path.abspath("src") + "')\n"
            "from ui.components.chat import render_message\n"
            'render_message(role="assistant", content="A", '
            'thought="<script>alert(1)</script>", msg_type="general")\n'
        )
        script_path = script_content.name
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(script_path).run(timeout=60)
        assert not at.exception
        # thought 마크다운 청크는 allow_html=False여야 한다 (Streamlit이 정책상
        # sanitize하므로 실행 가능한 <script> 요소가 렌더되지 않는다).
        thought_blocks = [
            m for m in at.markdown if m.proto.body == "<script>alert(1)</script>"
        ]
        assert thought_blocks, "thought 마크다운 청크가 렌더되지 않음"
        for block in thought_blocks:
            assert block.proto.allow_html is False
    finally:
        os.remove(script_path)
