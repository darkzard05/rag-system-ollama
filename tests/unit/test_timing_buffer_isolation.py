"""Per-query 타이밍 버퍼 격리 테스트 (Cross-session threading.local 오염 결함 수정 검증).

결함: ``_stage_timing_local = threading.local()``(구)이 동시 asyncio 쿼리가 단일
스레드에서 인터리빙될 때 세션 간 공유되어 ``[QUERY][TIMING]`` 스테이지 타이밍이
교차 오염됐습니다. ContextVar 기반으로 교체 후 다음을 검증합니다.

- (1) 동시 asyncio 태스크가 각자 자신의 타이밍 버퍼를 읽음 (격리)
- (2) 재시도(같은 태스크에서 노드 재실행) 시 누적이 보존됨
- (3) speculative ``ensure_future`` 태스크가 같은 dict를 공유하며 누적이 보존됨
- (4) ``[QUERY][TIMING]`` 로그 포맷이 불변 (벤치마크 파서 호환)
"""

import asyncio
import re
from unittest.mock import patch

import pytest

import core.graph._graph_core as gcore
import core.graph.graph_builder as gb


@pytest.mark.asyncio
async def test_concurrent_tasks_get_isolated_buffers():
    """동시 asyncio 태스크는 각자 자신의 타이밍 버퍼를 읽어야 합니다."""

    async def worker_a():
        gb._reset_stage_timings()
        gb._add_stage_ms("preprocess_ms", 10.0)
        await asyncio.sleep(0)  # yield so B resets its own buffer
        gb._add_stage_ms("retrieve_ms", 20.0)
        await asyncio.sleep(0)
        return dict(gb._stage_timing_var.get({}))

    async def worker_b():
        gb._reset_stage_timings()
        gb._add_stage_ms("preprocess_ms", 500.0)
        gb._add_stage_ms("grade_ms", 700.0)
        await asyncio.sleep(0)
        return dict(gb._stage_timing_var.get({}))

    a, b = await asyncio.gather(worker_a(), worker_b())

    assert a.get("preprocess_ms") == 10.0, (
        f"Task A가 자신의 preprocess(10.0)을 읽어야 하는데 {a.get('preprocess_ms')}를 읽음"
    )
    assert a.get("retrieve_ms") == 20.0, (
        f"Task A가 자신의 retrieve(20.0)을 읽어야 하는데 {a.get('retrieve_ms')}를 읽음"
    )
    assert b.get("preprocess_ms") == 500.0, (
        f"Task B가 자신의 preprocess(500.0)을 읽어야 하는데 {b.get('preprocess_ms')}를 읽음"
    )
    assert a.get("grade_ms", 0.0) == 0.0, (
        f"Task A는 B의 grade(700)로 오염되면 안 됨 — {a.get('grade_ms')}를 읽음"
    )


@pytest.mark.asyncio
async def test_retry_accumulation_preserved():
    """같은 태스크에서 노드 재실행 시 누적이 보존되어야 합니다."""
    gb._reset_stage_timings()
    gb._add_stage_ms("retrieve_ms", 100.0)
    gb._add_stage_ms("retrieve_ms", 50.0)  # 재시도 재실행 = 누적
    snap = dict(gb._stage_timing_var.get({}))
    assert snap.get("retrieve_ms") == 150.0, (
        f"재시도 누적(100+50=150)이 보존되어야 하는데 {snap.get('retrieve_ms')}"
    )


@pytest.mark.asyncio
async def test_speculative_task_shares_buffer_and_accumulates():
    """speculative ensure_future 태스크가 같은 dict를 공유하며 누적이 보존되어야 합니다."""
    gb._reset_stage_timings()
    gb._add_stage_ms("retrieve_ms", 100.0)

    async def spec_worker():
        await asyncio.sleep(0)
        gb._add_stage_ms("grade_ms", 50.0)
        await asyncio.sleep(0)

    task = asyncio.ensure_future(spec_worker())
    await asyncio.sleep(0)
    gb._add_stage_ms("grade_ms", 30.0)
    await task

    snap = dict(gb._stage_timing_var.get({}))
    assert snap.get("grade_ms") == 80.0, (
        f"speculative 누적(30+50=80)이 보존되어야 하는데 {snap.get('grade_ms')}"
    )
    assert snap.get("retrieve_ms") == 100.0, (
        f"retrieve 누적이 유지되어야 하는데 {snap.get('retrieve_ms')}"
    )


def test_query_timing_format_preserved():
    """[QUERY][TIMING] 로그 포맷이 벤치마크 파서와 호환되도록 보존되어야 합니다.

    실제 ``_emit_query_timing``을 호출해 포맷을 검증합니다. 이전 버전은 로그 문자열을
    테스트 안에서 재구성했기에 실제 함수와 drift가 생겨 계약을 못 지켰습니다.
    """
    captured: list[str] = []
    timings = {
        "preprocess_ms": 1.2,
        "retrieve_ms": 3.4,
        "grade_ms": 5.6,
        "generate_total_ms": 7.8,
        "ttft_ms": 0.9,
        "first_token_ms": 0.4,
    }
    with patch.object(gcore.logger, "info", side_effect=captured.append):
        gcore._emit_query_timing(timings)

    assert len(captured) == 1, captured
    line = captured[0]
    assert line.startswith("[QUERY][TIMING]"), line
    for key in (
        "preprocess_ms",
        "retrieve_ms",
        "grade_ms",
        "generate_total_ms",
        "ttft_ms",
        "first_token_ms",
    ):
        assert f"{key}=" in line, f"포맷에 {key}= 키가 없음: {line}"

    # bench_query_latency.py:79 정규식이 additive 키가 추가돼도 ttft_ms를 추출해야 한다.
    match = re.search(r"ttft_ms=(?P<ttft_ms>[\d.]+)", line)
    assert match is not None, line
    assert float(match.group("ttft_ms")) == 0.9, line
