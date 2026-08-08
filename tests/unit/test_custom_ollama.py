"""F2a extraction-key hardening tests for DeepThinkingChatOllama.

Verifies the thought extractor reads ``additional_kwargs["reasoning_content"]``
(the key langchain-ollama 0.3.10 writes, chat_models.py:971/1047) first, with
``reasoning``/``thinking``/``thought`` as legacy fallbacks.
"""

from langchain_core.messages import AIMessageChunk
from src.core.custom_ollama import DeepThinkingChatOllama


def test_reasoning_content_extracted_from_additional_kwargs():
    chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "rt"})
    content, thought = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )._convert_chunk_to_thought_and_content(chunk)
    assert content == ""
    assert thought == "rt"


def test_reasoning_content_fallback_without_content_blocks():
    # langchain-core 1.4.9 injects reasoning_content into content_blocks
    # (ai.py:295-302), so the v1 short-circuit keeps content_blocks empty
    # and forces the additional_kwargs fallback path under test.
    chunk = AIMessageChunk(
        content=[],
        additional_kwargs={"reasoning_content": "rt"},
        response_metadata={"output_version": "v1"},
    )
    content, thought = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )._convert_chunk_to_thought_and_content(chunk)
    assert content == ""
    assert thought == "rt"


def test_legacy_thinking_kwargs_fallback():
    chunk = AIMessageChunk(content="", additional_kwargs={"thinking": "t"})
    content, thought = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )._convert_chunk_to_thought_and_content(chunk)
    assert content == ""
    assert thought == "t"


def test_content_blocks_reasoning():
    chunk = AIMessageChunk(
        content="",
        content_blocks=[
            {"type": "reasoning", "reasoning": "rt"},
            {"type": "text", "text": "c"},
        ],
    )
    content, thought = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )._convert_chunk_to_thought_and_content(chunk)
    assert content == "c"
    assert thought == "rt"


def test_empty_chunk_returns_empty():
    chunk = AIMessageChunk(content="")
    content, thought = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )._convert_chunk_to_thought_and_content(chunk)
    assert content == ""
    assert thought == ""


def test_constructor_accepts_reasoning_kwarg():
    model = DeepThinkingChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M", reasoning=True
    )
    assert model.model == "qwen3:4b-instruct-2507-q4_K_M"
    assert model.reasoning is True
