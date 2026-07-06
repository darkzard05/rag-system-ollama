# Streaming Duplication Baseline Report

## Issue Summary
The RAG system currently emits two types of streaming events for the same content:
1. `on_chat_model_stream` (aliased as `messages`)
2. `on_custom_event` (aliased as `custom`)

When a `messages` event for the first token arrives slightly before the first `custom` event, the `StreamingResponseHandler` processes both, leading to duplication of the first token/word (e.g., "HelloHello world...").

## Reproduction Results
Using `scripts/verification/reproduce_streaming_duplication.py`:

| Scenario | Result | Status |
|----------|--------|--------|
| Custom delayed (10ms) | `HelloHello world...` | **Reproduced** |
| Perfect Sync | `HelloHello world...` | **Reproduced** |
| Messages delayed (20ms) | `Hello world...` | OK |

## Root Cause Analysis
- `rag_core.py` yields both events.
- `streaming_handler.py` uses a boolean flag `using_custom_channel` that only flips *after* the first `custom` event is processed.
- There is no look-ahead or deduplication for the transition phase.

## Verification Results
Using `scripts/verification/verify_streaming_fix.py`:

| Scenario | Result | Status |
|----------|--------|--------|
| Custom delayed (10ms) | `Hello world this is a test.` | **✅ Fixed** |
| Perfect Sync | `Hello world this is a test.` | **✅ Fixed** |
| Short tokens ('is', 'a') | `... this is a test.` | **✅ Intact** |

## Implemented 3-Tier Defense Strategy

### 1. Engine Layer (`src/core/rag_core.py`)
- **Node-Aware Filtering:** The engine now identifies the active LangGraph node. If the `generate` node is active, it automatically suppresses `on_chat_model_stream` events because they will be delivered via the superior `on_custom_event` channel.

### 2. Graph Layer (`src/core/graph_builder.py`)
- **Event Tagging:** Custom events in the `generate` node are now tagged with `source="dedicated_generator"`. This allows downstream handlers to know precisely which stream to trust.

### 3. Handler Layer (`src/api/streaming_handler.py`)
- **Channel Lock:** Once a custom event is received, the handler enters a "Custom Channel Lock" state, ignoring any subsequent standard message events for that session.
- **Refined Deduplication Window:** A tail-matching algorithm compares new tokens against the last 20 characters of processed text. It only omits tokens that are exact duplicates or very high-confidence overlaps, while protecting short common words (e.g., "is", "a", "the") from accidental deletion.

## Conclusion
The streaming duplication issue is resolved. The system now provides a stable, clean, and high-performance streaming experience even under unpredictable network or thread scheduling conditions.
