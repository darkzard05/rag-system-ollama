import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import asyncio
import orjson
from core.graph._graph_core import _recover_citations, _repair_json
from cache.coord_cache import coord_cache


def measure_json_recovery():
    blob = '{"final_answer": "test", "citations": [{"doc_id": "doc123", "text_span": "span", "section": "sec", "page": 1, "score": 0.95}]}'
    iterations = 1000
    start = time.perf_counter()
    cites = []
    for _ in range(iterations):
        cites = _recover_citations(blob)
    end = time.perf_counter()
    print(f"_recover_citations: {(end - start) * 1000:.2f} ms for {iterations} ops")

    start = time.perf_counter()
    for _ in range(iterations):
        _repair_json(blob)
    end = time.perf_counter()
    print(f"_repair_json: {(end - start) * 1000:.2f} ms for {iterations} ops")
    print(f"Count: {len(cites)}")


def measure_serialization():
    data = [
        {
            "doc_id": f"doc_{i}",
            "text_span": "span",
            "section": "sec",
            "page": 1,
            "score": 0.95,
        }
        for i in range(50)
    ]
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        orjson.dumps(data)
    end = time.perf_counter()
    print(f"orjson.dumps: {(end - start) * 1000:.2f} ms for {iterations} ops")


async def measure_coord_cache_async():
    iterations = 100
    start = time.perf_counter()
    for i in range(iterations):
        await coord_cache.get_coords(f"nonexistent_doc_{i}", 1)
    end = time.perf_counter()
    print(
        f"coord_cache.get_coords async (miss): {(end - start) * 1000:.2f} ms for {iterations} ops"
    )


if __name__ == "__main__":
    measure_json_recovery()
    measure_serialization()
    asyncio.run(measure_coord_cache_async())
