import asyncio
import time
import pytest
from fastapi import FastAPI
from api.api_server import lifespan


@pytest.mark.asyncio
async def test_lifespan_startup_does_not_block_health(monkeypatch):
    app = FastAPI()

    async def slow_warmup():
        await asyncio.sleep(5.0)

    monkeypatch.setattr("core.model_loader._warmup_models", slow_warmup)

    start_time = time.perf_counter()
    async with lifespan(app):
        startup_duration = time.perf_counter() - start_time
        assert startup_duration < 0.2, f"Startup blocked for {startup_duration:.3f}s"
