
import asyncio
import sys
import io
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.graph_builder import build_graph
from core.model_loader import load_llm
from common.config import DEFAULT_OLLAMA_MODEL
from langchain_core.runnables import RunnableConfig

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def repro_duplicate_output():
    print("🧪 [중복 출력 재현 테스트] 이벤트 수신 로직 정밀 분석")
    
    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    # 검색 없이 답변만 생성하는 빈 리트리버 그래프 생성
    app = build_graph() 
    config = {"configurable": {"llm": llm}}

    question = "Hello, say 'Test' only."
    
    full_response = ""
    event_log = []

    print(f"질문: {question}")
    print("이벤트 수신 중...")

    # 실제 src/ui/ui.py 의 로직을 그대로 재현
    async for event in app.astream_events({"input": question}, config=config, version="v2"):
        kind = event["event"]
        name = event.get("name", "Unknown")
        data = event.get("data", {})
        
        chunk_text = None
        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if hasattr(chunk, "content"): chunk_text = chunk.content
            elif isinstance(chunk, dict): chunk_text = chunk.get("content")
            if chunk_text:
                event_log.append(f"[ChatModel] {chunk_text}")
        
        elif kind == "on_parser_stream":
            chunk_text = data.get("chunk")
            if chunk_text:
                event_log.append(f"[Parser] {chunk_text}")
        
        if chunk_text:
            full_response += chunk_text

    print("\n--- 이벤트 수신 로그 (상위 10개) ---")
    for log in event_log[:10]:
        print(log)

    print("\n--- 최종 응답 결과 ---")
    print(f"결과값: '{full_response}'")

    # 검증: 같은 단어가 연속으로 나타나는지 확인 (예: 'TTee sst t')
    # 간단하게 글자 수가 비정상적으로 많은지 또는 동일 패턴 반복 확인
    is_duplicate = any(event_log[i].split() == event_log[i+1].split() 
                       for i in range(len(event_log)-1) 
                       if "[ChatModel]" in event_log[i] and "[Parser]" in event_log[i+1])
    
    if is_duplicate:
        print("\n❌ FAIL: 중복 출력이 감지되었습니다! ChatModel과 Parser 이벤트를 동시에 수신 중입니다.")
    else:
        print("\n✅ PASS: 중복 출력이 없습니다.")

if __name__ == "__main__":
    asyncio.run(repro_duplicate_output())
