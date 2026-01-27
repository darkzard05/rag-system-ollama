
import ollama

def test_qwen_thinking_attribute():
    print("\n" + "="*60)
    print("🧐 [속성 검증] 'thinking' 속성 존재 여부 확인")
    print("="*60)

    question = "방 안에 3명의 사람이 있고 서로 한 번씩 악수하면 총 몇 번일까? 단계별로 생각해서 답해줘."
    
    try:
        # 스트리밍 모드
        stream = ollama.generate(model='qwen3:4b-instruct-2507-q4_K_M', prompt=question, stream=True)
        
        thinking_found = False
        content_found = False
        
        for chunk in stream:
            # 1. 'thinking' 또는 'thought' 속성이 있는지 직접 검사
            # 최신 라이브러리 객체는 hasattr로 검사 가능
            thinking_val = getattr(chunk, 'thinking', None) or getattr(chunk, 'thought', None)
            
            if thinking_val:
                if not thinking_found:
                    print("\n[🧠 사고 시작]")
                    thinking_found = True
                print(thinking_val, end="", flush=True)
            
            # 2. 일반 답변 내용
            content_val = getattr(chunk, 'response', '')
            if content_val:
                if not content_found:
                    print("\n\n[📢 최종 답변]")
                    content_found = True
                print(content_val, end="", flush=True)
        
        print(f"\n\n[결과] thinking/thought 속성 감지: {thinking_found}")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")

if __name__ == "__main__":
    test_qwen_thinking_attribute()
