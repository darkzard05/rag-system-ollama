"""UI-owned runtime overrides for deployment defaults.

``config.yml`` keeps the startup defaults. The settings modal owns the live
value. A proxy object carries the default and resolves any override at use
time, so existing ``if FLAG:`` and ``FLAG == "value"`` call sites need no
change. Module-level patches in tests still work because they replace the
proxy with a plain value in the consuming module.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.RLock()
_overrides: dict[str, Any] = {}


class RuntimeSetting:
    """A config value whose current value can be changed while the app runs."""

    def __init__(
        self, key: str, default: Any, choices: tuple[Any, ...] | None = None
    ) -> None:
        self.key = key
        self._default = default
        self._choices = choices

    def get(self) -> Any:
        """Return the live override, or the startup default."""
        with _lock:
            return _overrides.get(self.key, self._default)

    def set(self, value: Any) -> None:
        """Validate and store a runtime override."""
        coerced = type(self._default)(value)
        if self._choices is not None and coerced not in self._choices:
            raise ValueError(
                f"{self.key} must be one of {self._choices}, got {coerced!r}"
            )
        with _lock:
            _overrides[self.key] = coerced

    def reset(self) -> None:
        """Drop the override and fall back to the startup default."""
        with _lock:
            _overrides.pop(self.key, None)

    def __bool__(self) -> bool:
        return bool(self.get())

    def __eq__(self, other: object) -> bool:
        return self.get() == other

    def __hash__(self) -> int:
        return hash(self.get())

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key}={self.get()!r})"

    def lower(self) -> str:
        """String compatibility for consumers that normalize the value."""
        return str(self.get()).lower()
