import ollama
import time

def test_qwen_direct():
    print("\n" + "="*60)
    print("🚀 [Ollama Direct Test] qwen3:4b 객체 속성 정밀 분석")
    print("="*60)

    # 사고를 강력하게 유도하는 질문
    question = "방금 말한 사과 문제를 다시 생각해보자. 사고 과정을 <thought> 태그 안에 넣어서 출력해줘."
    print(f"질문: {question}\n")
    
    try:
        full_text = ""
        thought_detected = False
        
        # 스트리밍 수신
        stream = ollama.generate(model='qwen3:4b', prompt=question, stream=True)
        
        for chunk in stream:
            # 1. 객체의 모든 속성 조사 (첫 번째 청크에서만)
            if not full_text and not thought_detected:
                attrs = [a for a in dir(chunk) if not a.startswith('_')]
                print(f"   [청크 객체 속성]: {attrs}")
            
            # 2. 가능한 필드에서 데이터 추출
            # 최신 Ollama 라이브러리 기준 속성 접근
            content = getattr(chunk, 'response', '')
            
            # 사고 과정 필드 후보들 조사
            thought_candidates = ['thought', 'reasoning', 'context']
            for cand in thought_candidates:
                val = getattr(chunk, cand, None)
                if val:
                    if not thought_detected:
                        print(f"\n[✨ {cand.upper()} 필드 발견!]")
                        thought_detected = True
                    print(val, end="", flush=True)
            
            # 3. 일반 콘텐츠 출력
            if content:
                print(content, end="", flush=True)
                full_text += content
                
        print(f"\n\n[결과] 최종 답변 길이: {len(full_text)}자")
        print(f"[결과] 사고 과정 필드 감지: {thought_detected}")
        
        # 4. 만약 필드가 없다면 텍스트 내에 포함되어 있는지 확인
        if not thought_detected:
            if "<thought>" in full_text or "생각" in full_text:
                print("[결과] 사고 과정이 일반 텍스트 내에 포함되어 있습니다.")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    print("="*60 + "\n")

if __name__ == "__main__":
    test_qwen_direct()