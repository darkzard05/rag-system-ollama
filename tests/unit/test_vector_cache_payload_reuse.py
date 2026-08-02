import orjson
from langchain_core.documents import Document

from cache.vector_cache import _build_cache_payloads


def test_build_cache_payloads_reuses_serialized_splits_for_identical_bm25_docs():
    docs = [Document(page_content="hello world", metadata={"page": 1})]

    payloads = _build_cache_payloads(docs, docs)

    assert payloads["doc_splits_payload"] == payloads["bm25_payload"]
    assert orjson.loads(payloads["doc_splits_payload"]) == [
        {"page_content": "hello world", "metadata": {"page": 1}}
    ]
