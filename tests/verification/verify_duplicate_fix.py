import asyncio
import io
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from common.config import DEFAULT_OLLAMA_MODEL
from core.graph_builder import build_graph
from core.model_loader import load_llm

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def verify_duplicate_fix_final():
    print("🧪 [최종 중복 출력 검증] 핵심 엔진 수정 후 테스트 시작")

    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    app = build_graph()
    config = {"configurable": {"llm": llm}}

    question = "Hello, say 'Test' only."
    full_response = ""

    # 현재 앱의 최신 로직 (on_parser_stream만 사용)
    async for event in app.astream_events(
        {"input": question}, config=config, version="v2"
    ):
        kind = event["event"]
        event.get("name", "Unknown")
        data = event.get("data", {})

        if kind == "on_parser_stream":
            chunk_text = data.get("chunk")
            if chunk_text:
                full_response += chunk_text

    print(f"\n결과값: '{full_response}'")

    # 모델에 따라 공백이 포함될 수 있으므로 strip()으로 비교
    if full_response.strip() == "Test":
        print("\n✅ PASS: 중복 출력이 완전히 해결되었습니다!")
    else:
        print(
            f"\n❌ FAIL: 결과가 여전히 중복되거나 잘못되었습니다. (결과: '{full_response}')"
        )


if __name__ == "__main__":
    asyncio.run(verify_duplicate_fix_final())
