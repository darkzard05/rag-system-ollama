"""Shared ``final_answer`` extraction from structured raw-JSON streams.

The UI and evaluation harness must agree on what the user actually sees. This
module owns that single definition: raw JSON chunks are scanned incrementally
and only the ``final_answer`` string value is emitted as display text.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "FinalAnswerExtractor",
    "extract_final_answer_delta",
    "recover_final_answer",
]

_ESCAPE_DECODE: dict[str, str] = {
    "n": "\n",
    '"': '"',
    "\\": "\\",
    "t": "\t",
    "/": "/",
}

_DELIMITERS = (":", ",", "}", "]")

_FA_RE = re.compile(r'"final_answer"\s*:\s*"(.*)', re.DOTALL)


def extract_final_answer_delta(
    buffer: str,
    start: int,
    key_pos: list[Any] | None = None,
) -> tuple[str, int]:
    """Incrementally pull the growing ``final_answer`` string value out of a
    partial JSON buffer. Returns ``(delta_text, new_scan_pos)``.

    O(1) amortized incremental scanner preserving all escape sequences
    (\\n, \\", \\uXXXX) and delimiter-aware close semantics.
    """
    n = len(buffer)
    key = '"final_answer"'

    # --- Incremental state / verified fast path ---
    # key_pos 구조: [key_idx, raw_pos, pending_escape, pending_u, closed, verified]
    # verified(6번째 슬롯)는 ``FinalAnswerExtractor``만 설정한다: 키/콜론/여는
    # 따옴표를 이미 located 했고, 이후 버퍼가 append-only 이거나 소비된 앞부분만
    # 잘라내는 tail-compaction 이라는 호출자 불변식을 보증한다. verified 면
    # Phase 1-3(불변 헤드 재검증)를 통째로 건너뛰고 raw_pos 부터 바로 스캔한다.
    # NOTE: 스킵 검사는 Phase 1-3 보다 먼저 수행되어야 한다 — 압축된 버퍼는
    # 헤드를 포함하지 않으므로 Phase 2/3 이 early-return 으로 오판한다.
    is_incremental = key_pos is not None and len(key_pos) >= 5
    use_verified_skip = (
        is_incremental
        and key_pos is not None
        and len(key_pos) >= 6
        and bool(key_pos[5])
    )

    if use_verified_skip:
        assert key_pos is not None  # use_verified_skip이 None 아님을 보장
        if key_pos[4]:  # 이미 닫힌 경우
            return "", start
        i = key_pos[1]
        pending_escape = bool(key_pos[2])
        pending_u = key_pos[3]
    else:
        # --- Phase 1: key search ---
        key_idx: int
        if key_pos is not None and len(key_pos) > 0 and key_pos[0] >= 0:
            key_idx = key_pos[0]
        else:
            key_idx = buffer.find(key, 0)
            if key_pos is not None and len(key_pos) > 0:
                key_pos[0] = key_idx
            if key_idx == -1:
                return "", 0

        # --- Phase 2: verify key is followed by optional ws + colon ---
        after_key = key_idx + len(key)
        j = after_key
        while j < n and buffer[j] in " \t\n\r":
            j += 1
        if j >= n or buffer[j] != ":":
            return "", start

        # --- Phase 3: find value's opening quote ---
        i = j + 1
        while i < n and buffer[i] in " \t\n\r":
            i += 1
        if i >= n or buffer[i] != '"':
            return "", start

        open_quote = i
        value_start = open_quote + 1

        # --- Phase 4: incremental value scan ---
        if is_incremental:
            assert key_pos is not None  # is_incremental이 None 아님을 보장
            if key_pos[4]:  # 이미 닫힌 경우
                return "", start
            i = key_pos[1] if key_pos[1] >= value_start else value_start
            pending_escape = bool(key_pos[2])
            pending_u = key_pos[3]
            if len(key_pos) >= 6:
                # 따옴표까지 located — 다음 호출부터 Phase 1-3 생략 가능.
                key_pos[5] = 1
        else:
            i = value_start
            pending_escape = False
            pending_u = None

    delta_chars: list[str] = []

    while i < n:
        ch = buffer[i]

        # Deferred escape from previous call
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
            delta_chars.append("\\")
            delta_chars.append(ch)
            i += 1
            continue

        # Deferred \uXXXX from previous call
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
                delta_chars.append(pending_u)
                pending_u = None
                continue

        # Normal characters
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
                delta_chars.append(ch)
                delta_chars.append(nxt)
                i += 2
                continue
            pending_escape = True
            i += 1
            continue

        if ch == '"':
            nxt = buffer[i + 1] if i + 1 < n else ""
            if nxt in _DELIMITERS or nxt == "":
                if is_incremental:
                    assert key_pos is not None  # is_incremental이 None 아님을 보장
                    key_pos[4] = 1  # closed = True
                if is_incremental:
                    new_delta = "".join(delta_chars)
                    return new_delta, start + len(new_delta)
                else:
                    new_delta = "".join(delta_chars)[start:]
                    return new_delta, len(delta_chars)
            delta_chars.append(ch)
            i += 1
            continue

        delta_chars.append(ch)
        i += 1

    if is_incremental:
        assert key_pos is not None  # is_incremental이 None 아님을 보장
        key_pos[1] = i
        key_pos[2] = 1 if pending_escape else 0
        key_pos[3] = pending_u
        new_delta = "".join(delta_chars)
        return new_delta, start + len(new_delta)
    else:
        new_delta = "".join(delta_chars)[start:]
        return new_delta, len(delta_chars)


def recover_final_answer(blob: str) -> str | None:
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


class FinalAnswerExtractor:
    """structured 모드 raw_json 청크 → final_answer 증분 추출 헬퍼 (O(n) total).

    ``_chunks`` 에 입력 전체를 append-only 로 보관하고, 스캐너에 넘기는
    ``_blob`` 은 스캔이 끝난 앞부분을 매 feed 마다 잘라내는 sliding window 로
    유지한다. 스캐너가 verified 이후 Phase 1-3 을 생략하므로 앞부분 삭제가
    안전하고, 매 feed 복사량은 미스캔 꼬리+새 청크로 상수가 된다. 기존의 매
    토큰 전체 복사(``_blob += content`` O(n²) total)를 제거한다.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._blob: str = ""
        self._scan_pos = 0
        # [key_idx, raw_pos, pending_escape, pending_u, closed, verified]
        self._key_pos: list[Any] = [-1, -1, 0, None, 0, 0]

    def feed(self, content: str) -> str:
        """raw_json 청크의 content를 누적하고 새 final_answer delta를 즉시 반환."""
        self._chunks.append(content)
        if self._key_pos[5] and self._key_pos[1] > 0:
            # 소비된 앞부분은 디코드 완료 — 스캐너 재개점만 남기고 버린다.
            # (pending_escape/pending_u 플래그가 경계 이스케이프를 carries)
            self._blob = self._blob[self._key_pos[1] :]
            self._key_pos[1] = 0
        self._blob += content
        delta, self._scan_pos = extract_final_answer_delta(
            self._blob, self._scan_pos, self._key_pos
        )
        return delta

    @property
    def closed(self) -> bool:
        """final_answer 값이 닫혔으면 True (이후 토큰은 citations/confidence)."""
        return bool(self._key_pos[4])
