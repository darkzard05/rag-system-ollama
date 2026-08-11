"""P4 regression: JsonPlusSerializer round-trips + warning-free instantiation.

Locks the serializer configuration from ``src/core/graph_builder.py:1008-1017``
(``JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[Human/AI/System])``,
applied in T9) so a future drift -- allowlist collapse, pickle fallback re-enable,
``allowed_objects`` wording change -- fails loudly.

Notes:
- ``BaseMessage`` subclasses serialize through the msgpack EXT_PYDANTIC_V2 path
  (``model_dump``/``model_validate_json``), so the LC-constructor allowlist check is
  bypassed for messages. ``ToolMessage`` (module ``langchain_core.messages.tool``)
  therefore round-trips even though it is NOT in the allowlist -- verified empirically
  in T10, so no graph_builder.py change was required.
- Importing ``langgraph.checkpoint.serde.jsonplus`` emits ONE
  ``LangChainPendingDeprecationWarning`` at module import time (``LC_REVIVER = Reviver()``
  at module top level). That import-time warning is the pre-existing T1 baseline and is
  NOT what this test asserts; the warning-absence test is scoped to instantiation only.
"""

import warnings

import ormsgpack
import pytest
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

_ALLOWED_JSON_MODULES: list[tuple[str, ...]] = [
    ("langchain_core", "messages", "HumanMessage"),
    ("langchain_core", "messages", "AIMessage"),
    ("langchain_core", "messages", "SystemMessage"),
]


def _serde() -> JsonPlusSerializer:
    """Construct the serializer exactly as src/core/graph_builder.py does (P4 block)."""
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=_ALLOWED_JSON_MODULES,
    )


def test_message_list_roundtrip_preserves_content() -> None:
    """Given: a message-list payload. When: dumps_typed -> loads_typed. Then: preserved."""
    serde = _serde()
    payload = {
        "messages": [HumanMessage("hi"), AIMessage("yo"), SystemMessage("sys")],
    }

    typed, data = serde.dumps_typed(payload)
    restored = serde.loads_typed((typed, data))

    assert typed == "msgpack"  # pickle fallback path unused
    assert [type(m) for m in restored["messages"]] == [
        HumanMessage,
        AIMessage,
        SystemMessage,
    ]
    assert [m.content for m in restored["messages"]] == ["hi", "yo", "sys"]


def test_tool_message_roundtrip_works_without_allowlist_entry() -> None:
    """ToolMessage round-trips via the pydantic-v2 ext path; no allowlist entry needed."""
    serde = _serde()
    tool_message = ToolMessage(content="tool result", tool_call_id="call_123")

    typed, data = serde.dumps_typed({"messages": [tool_message]})
    restored = serde.loads_typed((typed, data))

    restored_message = restored["messages"][0]
    assert type(restored_message) is ToolMessage
    assert restored_message.content == "tool result"
    assert restored_message.tool_call_id == "call_123"


def test_pure_payload_roundtrip_isomorphic_to_sanitized_state() -> None:
    """Pure primitives (the _sanitize_channel_value output shape) round-trip exactly."""
    serde = _serde()
    payload = {
        "question": "cm3가 뭔가요?",
        "docs": [{"page": 1, "title": "2201.07520v1"}, [], {}],
        "rerank_score": 0.85,
        "count": 3,
        "flag": True,
        "none": None,
    }

    typed, data = serde.dumps_typed(payload)

    assert serde.loads_typed((typed, data)) == payload


def test_pickle_fallback_disabled_rejects_unserializable_objects() -> None:
    """pickle_fallback=False must raise instead of silently downgrading to pickle."""
    serde = _serde()

    with pytest.raises(ormsgpack.MsgpackEncodeError):
        serde.dumps_typed({"callback": lambda: 1})
    with pytest.raises(NotImplementedError):
        serde.loads_typed(("pickle", b"crafted-pickle-payload"))


def test_instantiation_emits_no_pending_deprecation_warning() -> None:
    """Constructing the serializer must not raise/record LangChainPendingDeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", LangChainPendingDeprecationWarning)
        _serde()

    deprecations = [
        w for w in caught if issubclass(w.category, LangChainPendingDeprecationWarning)
    ]
    assert deprecations == []
