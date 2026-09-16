"""Streaming extractors — pure final_answer delta state machine.

Cohesive unit extracted from ``streaming.py``. No Streamlit, no
SessionManager, no I/O — pure functions/classes only so tests can import
without a Streamlit runtime. ``streaming.py`` re-exports every name.

Moved verbatim (no logic/markup changes):
- ``_ESCAPE_DECODE`` / ``_DELIMITERS`` / ``_FA_RE``
- ``_extract_final_answer_delta``
- ``_recover_final_answer``
- ``_FinalAnswerExtractor``
"""

from __future__ import annotations

import re

_ESCAPE_DECODE: dict[str, str] = {
    "n": "\n",
    '"': '"',
    "\\": "\\",
    "t": "\t",
    "/": "/",
}

_DELIMITERS = (":", ",", "}", "]")


def _extract_final_answer_delta(
    buffer: str,
    start: int,
    _key_pos: list[int] | None = None,
) -> tuple[str, int]:
    """Incrementally pull the growing ``final_answer`` string value out of a
    partial JSON buffer.  Returns ``(delta_text, new_scan_pos)``.

    State-machine scanner that honours escape sequences (decoded, not
    literal), delimiter-aware close, pending-escape defer-to-next-call
    semantics, and optional key-position caching via *``_key_pos``* (an
    in-place ``list[int]``).
    """
    n = len(buffer)
    key = '"final_answer"'

    # --- Phase 1: key search (the ONLY str.find in the implementation) ---
    key_idx: int
    if _key_pos is not None and _key_pos[0] >= 0:
        key_idx = _key_pos[0]
    else:
        key_idx = buffer.find(key, 0)
        if _key_pos is not None:
            _key_pos[0] = key_idx
        if key_idx == -1:
            return "", 0

    # --- Phase 2: verify key is followed by optional ws + colon ---
    after_key = key_idx + len(key)
    j = after_key
    while j < n and buffer[j] in " \t\n\r":
        j += 1
    if j >= n or buffer[j] != ":":
        return "", start

    # --- Phase 3: find value's opening quote (char-by-char after colon) ---
    i = j + 1
    while i < n and buffer[i] in " \t\n\r":
        i += 1
    if i >= n or buffer[i] != '"':
        return "", start

    open_quote = i
    value_start = open_quote + 1

    # ``start`` counts decoded value chars already emitted (not buffer offset).
    emit_from = start

    # --- Phase 4: value scan with escape decode ---
    i = value_start
    delta_chars: list[str] = []
    pending_escape = False
    pending_u: str | None = None  # e.g. "\\u00" – holds incomplete \\uXXXX

    while i < n:
        ch = buffer[i]

        # Deferred escape from previous call --------------------------------
        if pending_escape:
            pending_escape = False
            decoded = _ESCAPE_DECODE.get(ch)
            if decoded is not None:
                delta_chars.append(decoded)
                i += 1
                continue
            if ch == "u":
                pending_u = "\\u"
                i += 1
                continue
            # Unknown escape: emit both chars literally.
            delta_chars.append("\\")
            delta_chars.append(ch)
            i += 1
            continue

        # Deferred \\uXXXX from previous call -------------------------------
        if pending_u is not None:
            if ch in "0123456789abcdefABCDEF" and len(pending_u) < 6:
                pending_u += ch
                i += 1
                if len(pending_u) == 6:
                    hex_str = pending_u[2:]
                    try:
                        delta_chars.append(chr(int(hex_str, 16)))
                    except ValueError:
                        delta_chars.append(pending_u)
                    pending_u = None
                continue
            else:
                # Non-hex terminates \\u: emit literal and reprocess ch.
                delta_chars.append(pending_u)
                pending_u = None
                continue

        # Normal characters --------------------------------------------------
        if ch == "\\":
            if i + 1 < n:
                nxt = buffer[i + 1]
                decoded = _ESCAPE_DECODE.get(nxt)
                if decoded is not None:
                    delta_chars.append(decoded)
                    i += 2
                    continue
                if nxt == "u":
                    pending_u = "\\u"
                    i += 2
                    continue
                # Unknown escape: emit both literally.
                delta_chars.append(ch)
                delta_chars.append(nxt)
                i += 2
                continue
            # Trailing backslash at end-of-buffer: defer.
            pending_escape = True
            i += 1
            continue

        if ch == '"':
            # Delimiter-aware close: value terminates only when the next char
            # is a JSON delimiter or the buffer is exhausted.
            nxt = buffer[i + 1] if i + 1 < n else ""
            if nxt in _DELIMITERS or nxt == "":
                new_delta = "".join(delta_chars)[emit_from:]
                return new_delta, len(delta_chars)
            # Unescaped inner quote – treat as content.
            delta_chars.append(ch)
            i += 1
            continue

        delta_chars.append(ch)
        i += 1

    # Value still open – emit only chars not yet delivered; pending escape/u
    # are held for the next call and NOT emitted in this delta.
    new_delta = "".join(delta_chars)[emit_from:]
    return new_delta, len(delta_chars)


_FA_RE = re.compile(r'"final_answer"\s*:\s*"(.*)', re.DOTALL)


def _recover_final_answer(blob: str) -> str | None:
    """깨진 JSON에서 final_answer 값을 정규식으로 복구한다.

    스트리밍 중 누적된 raw_json 이 닫히지 않은 따옴표/이스케이프로 인해
    json.loads 에 실패하더라도, 버블에는 원시 JSON 이 아닌 복구된 정답 텍스트만
    남도록 한다. 복구 불가능하면 None 반환.
    """
    if not blob:
        return None
    m = _FA_RE.search(blob)
    if not m:
        return None
    value = m.group(1)
    # 닫는 따옴표가 있으면 그 전까지, 없으면 그대로(열린 값) 사용
    end = value.find('"')
    if end != -1:
        value = value[:end]
    return value.strip()


class _FinalAnswerExtractor:
    """structured 모드 raw_json 청크 → final_answer 증분 추출 헬퍼.

    consume_stream_into_message / _content_generator / stream_content 세
    소비자가 공유한다. feed(content)마다 내부 blob에 원문을 순서대로
    누적하고, _extract_final_answer_delta의 상태 머신(스캔 위치 + 키 위치
    캐시)으로 새로 디코드된 delta만 반환한다. 모든 delta의 연결(join)은
    final_answer 값과 정확히 같다(순서 불변, \\uXXXX/CJK 디코드 포함).
    스트림/제너레이터당 인스턴스 1개를 만들고 chunk마다 feed() 호출.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._scan_pos = 0
        self._key_pos: list[int] = [-1]

    def feed(self, content: str) -> str:
        """raw_json 청크의 content를 누적하고 새 final_answer delta를 반환.

        키 미발견 등 아직 출력할 문자가 없으면 ""를 반환한다. 호출자는
        yield 직전에 ``if delta:`` 가드로 빈 문자열을 걸러야 한다.
        """
        self._parts.append(content)
        blob = "".join(self._parts)
        delta, self._scan_pos = _extract_final_answer_delta(
            blob, self._scan_pos, self._key_pos
        )
        return delta
