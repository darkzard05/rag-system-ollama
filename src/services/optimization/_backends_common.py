"""
Cache backends common - shared base layer.

Extracted from _backends.py (Batch 5.1). Holds shared constants,
type vars, _json_default, CacheEntry, CacheBackend. Other backend
modules depend on this; it imports nothing from sibling backends
(no cycles).
"""

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Generic, TypeVar

import numpy as np

from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)

# type vars
T = TypeVar("T")
R = TypeVar("R")

# ---------------------------------------------------------------
# cache serialization format
# ---------------------------------------------------------------
CACHE_FORMAT = "json-v2"
CACHE_FORMAT_KEY = "_fmt"


def _json_default(obj: Any) -> Any:
    """json.dump 의 default 인코더.

    지원 타입을 명시적으로 변환하고, 변환 불가능한 타입은
    조용히 문자열화하지 않고 TypeError 를 발생시켜
    무음 손상을 방지한다.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, set | frozenset):
        return list(obj)
    try:
        from pydantic import BaseModel

        if isinstance(obj, BaseModel):
            return obj.model_dump()
    except ImportError:
        pass
    try:
        # numpy scalar -> python 네이티브
        if isinstance(obj, np.generic):
            return obj.item()
        # numpy array -> list (pickle 은 ndarray 를 그대로 직렬화했으므로 동등 보장)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    msg = f"JSON 으로 직렬화할 수 없는 타입입니다: {type(obj).__name__}"
    raise TypeError(msg)


@dataclass
class CacheEntry:
    """캐시 항목"""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: float
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """TTL 만료 여부 확인"""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def get_age(self) -> float:
        """항목 나이 (초)"""
        return time.time() - self.created_at

    def touch(self) -> None:
        """접근 시간 업데이트"""
        self.accessed_at = time.time()
        self.hit_count += 1

    def to_json_dict(self) -> dict[str, Any]:
        """안전한 JSON 직렬화용 dict 로 변환 (unsafe 역직렬화 대체)."""
        return {
            CACHE_FORMAT_KEY: CACHE_FORMAT,
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """JSON dict → CacheEntry 복원. 포맷 불일치 시 ValueError."""
        if data.get(CACHE_FORMAT_KEY) != CACHE_FORMAT:
            msg = (
                f"지원되지 않는 캐시 포맷입니다: "
                f"{data.get(CACHE_FORMAT_KEY)!r} (expected {CACHE_FORMAT!r})"
            )
            raise ValueError(msg)
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            accessed_at=data["accessed_at"],
            ttl_seconds=data["ttl_seconds"],
            hit_count=data.get("hit_count", 0),
            metadata=data.get("metadata", {}),
        )


class CacheBackend(ABC, Generic[T]):
    """캐시 백엔드 추상 클래스"""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """값 조회"""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """값 삭제"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """캐시 전체 삭제"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        pass
