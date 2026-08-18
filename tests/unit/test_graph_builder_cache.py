"""
Phase 3 회귀 테스트: build_graph()는 프로세스 전역 캐시를 반환한다.

실제 근본 원인(과장 수정): build_graph()는 _graph_cache.compiled(프로세스 전역)를
반환하므로 세션 퇴출로 엔진이 휘발되지 않음. 즉 동일 프로세스 내 재컴파일은 0회.
본 테스트는 동일 프로세스에서 build_graph()를 2회 호출해 동일 객체(캐시 히트)를
반환함을 검증한다. (프로세스 재시작 시 재컴파일은 컴파일된 LangGraph 직렬화 불가로
수용 가능 — 문서화됨.)
"""

import asyncio

from core.graph_builder import build_graph, invalidate_graph_cache


def test_build_graph_returns_process_global_cached_object() -> None:
    """동일 프로세스 내 2회 호출이 동일 컴파일 객체를 반환(재컴파일 0회)."""
    invalidate_graph_cache()
    try:
        first = asyncio.run(build_graph())
        second = asyncio.run(build_graph())
        assert first is second, "build_graph()는 프로세스 전역 캐시를 반환해야 함"
        assert first is not None
    finally:
        invalidate_graph_cache()
