"""Incremental scanner for the ``citations`` array in streamed JSON.

Dispatches completed citation objects mid-generation so the UI can render
references before the full JSON document finishes streaming. The scanner is
stateful: ``_cursor``/``_emitted`` persist across ``feed`` calls, and a fresh
instance must be created per generation attempt.
"""

from __future__ import annotations

import json
from typing import Any

_SUICIDE_BYTES = 64 * 1024

_SEEK_KEY = "SEEK_KEY"
_SEEK_ARRAY = "SEEK_ARRAY"
_IN_ARRAY = "IN_ARRAY"
_IN_OBJECT = "IN_OBJECT"
_IN_STRING = "IN_STRING"
_ESCAPE = "ESCAPE"
_UNICODE_ESCAPE = "UNICODE_ESCAPE"
_DONE = "DONE"


class PartialCitationScanner:
    """Stateful, order-agnostic scanner over the streamed JSON buffer."""

    def __init__(self) -> None:
        self._state = _SEEK_KEY
        self._cursor = 0
        self._emitted = 0
        self._array_start: int | None = None
        self._obj_start = 0
        self._depth = 0
        self._string_return = _IN_OBJECT
        self._unicode_remaining = 0
        self.dead = False

    @property
    def emitted(self) -> int:
        """Number of citation objects harvested so far."""
        return self._emitted

    def feed(self, accumulated: str) -> list[tuple[int, dict[str, Any]]]:
        """Scan ``accumulated`` and return newly completed citation objects.

        Returns a list of ``(index, object)`` tuples where ``index`` is the
        harvest order within the citations array. Objects that fail
        ``json.loads`` validation stay buffered (not emitted). Once ``dead``
        is set (64KB past array start with zero harvest), always returns [].
        """
        if self.dead or self._state == _DONE:
            return []
        out: list[tuple[int, dict[str, Any]]] = []
        n = len(accumulated)
        while self._cursor < n:
            ch = accumulated[self._cursor]
            if self._state == _SEEK_KEY:
                idx = accumulated.find('"citations"', self._cursor)
                if idx < 0:
                    # Marker may straddle the buffer tail (partial marker at
                    # the end): rewind so the next feed re-examines the tail
                    # instead of skipping past the marker start forever.
                    self._cursor = max(0, n - len('"citations"') + 1)
                    break
                self._cursor = idx + len('"citations"')
                self._state = _SEEK_ARRAY
            elif self._state == _SEEK_ARRAY:
                if ch == "[":
                    self._array_start = self._cursor
                    self._cursor += 1
                    self._state = _IN_ARRAY
                elif ch in " \t\r\n:":
                    self._cursor += 1
                else:
                    # Marker found but no array follows (e.g. null) — give up.
                    self._state = _DONE
                    break
            elif self._state == _IN_ARRAY:
                if ch in " \t\r\n,":
                    self._cursor += 1
                elif ch == "]":
                    self._state = _DONE
                    self._cursor += 1
                elif ch == "{":
                    self._obj_start = self._cursor
                    self._depth = 1
                    self._cursor += 1
                    self._state = _IN_OBJECT
                else:
                    self._cursor += 1
            elif self._state == _IN_OBJECT:
                if ch == '"':
                    self._string_return = _IN_OBJECT
                    self._state = _IN_STRING
                    self._cursor += 1
                elif ch == "{":
                    self._depth += 1
                    self._cursor += 1
                elif ch == "}":
                    self._depth -= 1
                    self._cursor += 1
                    if self._depth == 0:
                        candidate = accumulated[self._obj_start : self._cursor]
                        try:
                            obj = json.loads(candidate)
                        except (json.JSONDecodeError, ValueError):
                            obj = None
                        if isinstance(obj, dict):
                            out.append((self._emitted, obj))
                            self._emitted += 1
                        self._state = _IN_ARRAY
                else:
                    self._cursor += 1
            elif self._state == _IN_STRING:
                if ch == "\\":
                    self._state = _ESCAPE
                    self._cursor += 1
                elif ch == '"':
                    self._state = self._string_return
                    self._cursor += 1
                else:
                    self._cursor += 1
            elif self._state == _ESCAPE:
                if ch == "u":
                    self._unicode_remaining = 4
                    self._state = _UNICODE_ESCAPE
                else:
                    self._state = _IN_STRING
                self._cursor += 1
            elif self._state == _UNICODE_ESCAPE:
                self._unicode_remaining -= 1
                self._cursor += 1
                if self._unicode_remaining == 0:
                    self._state = _IN_STRING
            else:
                break
            if (
                self._array_start is not None
                and self._emitted == 0
                and self._cursor - self._array_start > _SUICIDE_BYTES
            ):
                self.dead = True
                return []
        return out
