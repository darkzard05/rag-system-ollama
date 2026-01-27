
import asyncio
import time
import sys
import io
import html
from pathlib import Path
from unittest.mock import MagicMock

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.utils import apply_tooltips_to_response
from langchain_core.documents import Document

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_citation_rendering_integrity():
    print("🧪 [렌더링 테스트 1] 인용구 및 툴팁 변환 무결성 검증")
    
    mock_docs = [
        Document(page_content="This is content from page 1.", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="Special characters: < > & \" ' \nNew line here.", metadata={"page": 2, "source": "test.pdf"})
    ]
    
    test_cases = [
        {
            "input": "According to [p.1], the sky is blue.",
            "expected_contain": ['class="tooltip"', '[p.1]', 'This is content from page 1.'],
            "desc": "표준 인용구 [p.1] 변환"
        },
        {
            "input": "See (p. 2) for details.",
            "expected_contain": ['class="tooltip"', '[p.2]', 'Special characters: &lt; &gt; &amp;'],
            "desc": "괄호 형태 (p. 2) 및 HTML 이스케이프 확인"
        },
        {
            "input": "Information on [page 1] and [P.2].",
            "expected_contain": ['[p.1]', '[p.2]'],
            "desc": "다양한 대소문자 및 키워드 [page X], [P.X] 확인"
        }
    ]
    
    for case in test_cases:
        result = apply_tooltips_to_response(case["input"], mock_docs)
        passed = all(word in result for word in case["expected_contain"])
        print(f" - {case['desc']}: {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            print(f"   출력결과: {result}")

async def simulate_ui_throttling():
    print("\n🧪 [렌더링 테스트 2] UI 스트리밍 쓰로틀링(0.03s) 효율성 시뮬레이션")
    
    # 설정
    total_chunks = 100
    chunk_interval = 0.01 # 10ms 마다 토큰 도착 (매우 빠른 속도)
    throttling_period = 0.03 # UI 갱신 주기
    
    last_ui_update_time = 0
    ui_update_count = 0
    start_time = time.time()
    
    full_response = ""
    
    print(f"시뮬레이션 시작: 총 {total_chunks} 청크, 생성 간격 {chunk_interval*1000}ms, UI 주기 {throttling_period*1000}ms")
    
    for i in range(total_chunks):
        # 1. 청크 도착 시뮬레이션
        chunk_text = f"token_{i} "
        full_response += chunk_text
        await asyncio.sleep(chunk_interval)
        
        # 2. UI 렌더링 로직 (src/ui/ui.py 의 _stream_chat_response 로직 모사)
        current_time = time.time()
        if i == 0 or (current_time - last_ui_update_time > throttling_period):
            # 실제로는 answer_container.markdown(full_response + "▌") 호출
            ui_update_count += 1
            last_ui_update_time = current_time
            # UI 업데이트 부하 시뮬레이션
            await asyncio.sleep(0.005) 
            
    total_duration = time.time() - start_time
    reduction = (1 - (ui_update_count / total_chunks)) * 100
    
    print(f"결과 리포트:")
    print(f" - 총 수신 청크: {total_chunks}")
    print(f" - 실제 UI 갱신 횟수: {ui_update_count}")
    print(f" - 리프레시 감소율: {reduction:.1f}%")
    print(f" - 전체 소요 시간: {total_duration:.2f}s")
    
    if ui_update_count < total_chunks / 2:
        print("✅ PASS: 쓰로틀링이 효과적으로 작동하여 UI 부하를 줄였습니다.")
    else:
        print("❌ FAIL: 쓰로틀링이 제대로 작동하지 않았습니다.")

def test_markdown_edge_cases():
    print("\n🧪 [렌더링 테스트 3] 마크다운 엣지 케이스 확인")
    
    mock_docs = [Document(page_content="Table data", metadata={"page": 1})]
    
    # 표(Table) 문법과 인용구가 섞인 경우
    input_text = "| Header |\n| --- |\n| Data [p.1] |"
    result = apply_tooltips_to_response(input_text, mock_docs)
    
    # 인용구가 HTML로 변환되어도 표 문법이 깨지지 않는지 (시각적 확인 필요하나 여기선 패턴 확인)
    has_html = 'class="tooltip"' in result
    has_table_pipe = "|" in result
    
    print(f" - 표 내부 인용구 처리: {'✅ PASS' if has_html and has_table_pipe else '❌ FAIL'}")

if __name__ == "__main__":
    test_citation_rendering_integrity()
    asyncio.run(simulate_ui_throttling())
    test_markdown_edge_cases()
