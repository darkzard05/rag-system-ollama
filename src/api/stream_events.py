"""스트리밍 이벤트 매핑 모듈 — 스트리밍 파이프라인의 단일 진실 소스 (Single Source of Truth).

StreamChunk을 여섯 가지 표준 이벤트 타입으로 변환합니다.
SSE 라우트(`routes_chat.py`)와 UI 직접 경로(`streaming.py`)가
동일한 매핑을 공유하여 이벤트 구조가 동기 drifting하지 않도록 보장합니다.

이벤트 타입 (canonical order):
  1. "status"     — 노드 상태 메시지
  2. "message"    — LLM 응답 콘텐츠
  3. "thought"    — 모델 사고 과정
  4. "sources"    — 검색된 원본 문서
  5. "citations"  — 구조화된 인용 정보
  6. "metrics"    — 성능 통계
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.streaming_handler import StreamChunk

# ---------------------------------------------------------------------------
# Canonical event type registry — keep in sync with StreamEvent type field.
# Contract tests validate this tuple matches the six mapping branches below.
# ---------------------------------------------------------------------------
STREAM_EVENT_TYPES: tuple[str, ...] = (
    "status",
    "message",
    "thought",
    "sources",
    "citations",
    "metrics",
)


@dataclass(frozen=True)
class StreamEvent:
    """불변 스트리밍 이벤트 컨테이너.

    Attributes:
        type: 이벤트 종류 (STREAM_EVENT_TYPES 중 하나).
        payload: 이벤트별 데이터 딕셔너리.
    """

    type: str
    payload: dict[str, Any]


def chunk_to_stream_events(chunk: StreamChunk) -> list[StreamEvent]:
    """StreamChunk를 순서가 보장된 StreamEvent 목록으로 변환합니다.

    각 필드는 독립적으로 검사되며, 하나의 Chunk에서 여러 이벤트가
    생성될 수 있습니다. 반환 순서는 항상 1→6 (status→metrics)으로 고정됩니다.

    빈 값이나 누락된 필드는 해당 이벤트를 생성하지 않습니다.

    Args:
        chunk: 변환할 스트리밍 청크.

    Returns:
        이벤트 목록 (빈 청크의 경우 빈 리스트).
    """
    events: list[StreamEvent] = []

    # 모든 필드는 getattr로 방어적으로 읽는다: 테스트는 물론,
    # 일부 필드만 가진 duck-typed 청크(SimpleNamespace 등)도
    # AttributeError 없이 처리되어야 한다 (기존 소비자 방어 패턴과 동일).
    status = getattr(chunk, "status", None)
    node_name = getattr(chunk, "node_name", None)
    content = getattr(chunk, "content", None)
    thought = getattr(chunk, "thought", None)
    metadata = getattr(chunk, "metadata", None)
    citations = getattr(chunk, "citations", None)
    performance = getattr(chunk, "performance", None)

    # 1. status — 노드 상태 메시지
    if status:
        events.append(StreamEvent("status", {"message": status, "node": node_name}))

    # 2. message — LLM 응답 콘텐츠
    if content:
        events.append(StreamEvent("message", {"content": content}))

    # 3. thought — 모델 사고 과정
    if thought:
        events.append(StreamEvent("thought", {"content": thought}))

    # 4. sources — 원본 문서 (자르기 없이 전달)
    if metadata and "documents" in metadata:
        events.append(StreamEvent("sources", {"documents": metadata["documents"]}))

    # 5. citations — 구조화된 인용
    if citations:
        events.append(StreamEvent("citations", {"citations": citations}))

    # 6. metrics — 성능 통계
    if performance:
        events.append(StreamEvent("metrics", {"metrics": performance}))

    return events
