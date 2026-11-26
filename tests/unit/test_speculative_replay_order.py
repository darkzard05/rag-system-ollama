"""A late speculative event must not overtake buffered events during adoption."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from core.graph._generate import generate
from core.graph._graph_core import _dispatch_event
from core.graph._speculative_gen import (
    _current_spec,
    _spec_registry,
    _SpecEvent,
    _SpecGenerate,
)


@pytest.mark.asyncio
async def test_adoption_replays_buffer_in_emission_order() -> None:
    """Events emitted after adoption stay behind the events already buffered."""
    sid = "spec_replay_order_session"
    buffer = [
        _SpecEvent(name="E1", data={}, config={}),
        _SpecEvent(name="E2", data={}, config={}),
    ]
    sent: list[str] = []
    holder: dict[str, _SpecGenerate] = {}

    async def _late_spec_task() -> dict[str, str]:
        token = _current_spec.set(holder["spec"])
        try:
            await _dispatch_event("LATE", {}, writer=None, config={})
        finally:
            _current_spec.reset(token)
        return {"response": "ok"}

    async def _record(name: str, data: dict, config=None) -> None:  # noqa: ARG001
        sent.append(name)
        await asyncio.sleep(0)

    task = asyncio.create_task(_late_spec_task())
    holder["spec"] = _SpecGenerate(
        task=task, buffer=buffer, created_at=time.monotonic()
    )
    _spec_registry[sid] = holder["spec"]

    try:
        with (
            patch("core.graph._generate.adispatch_custom_event", new=_record),
            patch("core.graph._graph_core.adispatch_custom_event", new=_record),
        ):
            result = await generate(
                {"input": "질문"},
                {"configurable": {"thread_id": sid}},
                writer=MagicMock(),
            )
    finally:
        _spec_registry.pop(sid, None)
        if not task.done():
            task.cancel()

    assert result["response"] == "ok"
    assert sent == ["E1", "E2", "LATE"]
