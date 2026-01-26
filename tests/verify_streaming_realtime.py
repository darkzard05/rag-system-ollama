import asyncio
import sys
import io
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.graph_builder import build_graph
from core.model_loader import load_llm
from common.config import OLLAMA_MODEL_NAME

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def verify_streaming_realtime_final():
    print("🧪 [최종 실시간성 검증] 커스텀 이벤트 통로를 통한 실시간 전송 확인")
    
    llm = load_llm(OLLAMA_MODEL_NAME)
    app = build_graph() 
    config = {"configurable": {"llm": llm}}

    # 모델이 여러 토큰을 내뱉도록 질문 설정
    question = "Count from 1 to 5 slowly."
    
    full_response = ""
    render_steps = []
    start_time = time.time()

    print(f"질문: {question}")
    
    async for event in app.astream_events({"input": question}, config=config, version="v2"):
        kind = event["event"]
        name = event.get("name", "Unknown")
        data = event.get("data", {})
        
        chunk_text = None
        thought_text = None
        # [실제 앱 최신 로직] 커스텀 이벤트만 수신
        if kind == "on_custom_event" and name == "response_chunk":
            chunk_text = data.get("chunk")
            thought_text = data.get("thought")
        
        if thought_text:
            elapsed = time.time() - start_time
            print(f"🧠 사고 과정 수신: '{thought_text}'")
            render_steps.append(f"[{elapsed:.2f}s] THOUGHT: '{thought_text}'")

        if chunk_text:
            full_response += chunk_text
            elapsed = time.time() - start_time
            render_steps.append(f"[{elapsed:.2f}s] Chunk: '{chunk_text}' | Current: '{full_response}'")
            print(f"📍 실시간 조각 수신: '{chunk_text}'")

    print("\n--- 스트리밍 렌더링 타임라인 ---")
    for step in render_steps:
        print(step)

    print(f"\n최종 결과: '{full_response}'")
    
    # 검증: 조각이 여러 번에 걸쳐서 왔는가? (실시간성)
    is_realtime = len(render_steps) > 1
    # 중복이 없는가? (예: '11 22'가 아닌 '1 2')
    no_duplicate = "11" not in full_response and "22" not in full_response
    
    if is_realtime and no_duplicate:
        print("\n✅ PASS: 스트리밍이 실시간으로 중복 없이 완벽하게 작동합니다!")
    else:
        if not is_realtime: print("\n❌ FAIL: 답변이 한꺼번에 출력되었습니다.")
        if not no_duplicate: print("\n❌ FAIL: 여전히 중복 출력이 발생합니다.")

if __name__ == "__main__":
    asyncio.run(verify_streaming_realtime_final())