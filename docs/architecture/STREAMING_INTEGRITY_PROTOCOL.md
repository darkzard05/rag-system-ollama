# 📋 RAG 시스템 스트리밍 및 출력 무결성 프로토콜 (Streaming & Integrity Protocol)

본 문서는 답변 생성 시 발생하는 **중복 출력(Double Printing)** 및 **스트리밍 지연(Buffering)** 문제를 방지하기 위한 기술적 표준과 구현 규칙을 정의합니다.

---

## 1. 주요 발생 문제 및 원인 분석

### 1.1 중복 출력 (Double Printing)
- **증상**: 동일한 답변 글자가 두 번씩 반복해서 출력됨 (예: "안안녕녕하하세세요요").
- **원인**: LangChain의 `astream_events` 사용 시 `on_chat_model_stream`과 `on_parser_stream` 이벤트가 동일한 토큰을 각각 별도로 발생시키기 때문입니다. UI에서 두 통로를 모두 수신하면 중복이 발생합니다.

### 1.2 스트리밍 지연 (Blocky Output)
- **증상**: 답변이 실시간으로 나오지 않고, 한참 기다린 후 한꺼번에 "턱" 하고 출력됨.
- **원인**: LangGraph 노드 내부에서 토큰을 수동으로 소비(`_consume_stream`)하면, 해당 노드가 완전히 종료되기 전까지 표준 이벤트가 상위 루프로 전파되지 않고 노드 내부에 "갇히게" 됩니다.

---

## 2. 표준 아키텍처 (The Golden Rule)

현재 시스템은 **"전용 커스텀 통로 단일화"** 전략을 통해 두 문제를 동시에 해결합니다.

### 규칙 1: Core 엔진은 전용 이벤트를 발송한다
- `src/core/graph_builder.py`의 `generate_response` 노드 내에서 토큰이 생성될 때마다 `adispatch_custom_event`를 통해 **`response_chunk`**라는 이름의 커스텀 이벤트를 명시적으로 발송합니다.
- **이유**: 노드 내부 루프에서 직접 이벤트를 쏘면 노드 종료를 기다리지 않고 상위 루프로 즉시 전파됩니다 (실시간성 확보).

### 규칙 2: UI는 전용 이벤트만 수신한다
- `src/ui/ui.py`의 수신 루프는 오직 `on_custom_event` 중 이름이 `response_chunk`인 것만 필터링하여 `full_response`에 더합니다.
- **이유**: 표준 이벤트(`on_chat_model_stream` 등)를 의도적으로 무시함으로써 중복 발생 가능성을 0%로 차단합니다.

---

## 3. 핵심 코드 레퍼런스

### Core (발생부: `src/core/graph_builder.py`)
```python
# llm.astream 루프 내에서 조각이 생성될 때마다 즉시 발송
async for chunk in llm.astream(messages, config=config):
    msg = getattr(chunk, "message", chunk)
    content = msg.content
    thinking = msg.additional_kwargs.get("thinking", "")
    
    if content or thinking:
        await adispatch_custom_event(
            "response_chunk",
            {"chunk": content, "thought": thinking},
            config=config,
        )
```

### UI (수신부: `src/ui/ui.py`)
```python
# 다른 이벤트는 무시하고 오직 response_chunk만 수집함
if kind == "on_custom_event" and name == "response_chunk":
    chunk_text = data.get("chunk")
    
if chunk_text:
    full_response += chunk_text
    # 이후 렌더링 로직 수행
```

---

## 4. 금지 사항 (Anti-Patterns)

1.  ❌ **중복 수신 금지**: UI 코드에서 `on_parser_stream`과 `on_custom_event`를 동시에 수신하지 마십시오.
2.  ❌ **노드 내부 블로킹 금지**: `generate_response` 노드 내에서 `adispatch_custom_event` 없이 루프만 돌리면 스트리밍이 끊깁니다.
3.  ❌ **무분별한 sleep 제거**: 스트리밍 루프 내에 불필요한 `asyncio.sleep(0.01)` 등을 넣지 마십시오. 이미 0.03초 쓰로틀링이 UI 렌더링 부하를 관리하고 있습니다.

---

## 5. 검증 방법
기능 수정 후에는 반드시 아래 테스트를 실행하여 무결성을 확인하십시오.
- `python tests/verify_streaming_realtime.py`: 실시간 타임라인 및 중복 여부 체크.
