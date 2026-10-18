import json
from core.graph._graph_core import _repair_json


def test_json_repair_quotes() -> None:
    raw = '{"final_answer": "ok", "citations": [{"doc_id": "6daa326dee4d224d", "anchor": "test"}]}'
    repaired = _repair_json(raw)
    assert repaired is not None
    parsed = json.loads(repaired)
    assert parsed["citations"][0]["doc_id"] == "6daa326dee4d224d"
    assert parsed["citations"][0]["anchor"] == "test"
