import ollama
import time

def diagnose_and_retry():
    print("🔍 [Ollama Library] 인식 가능한 모델 목록 조회 중...")
    try:
        model_list = ollama.list()
        actual_names = []
        
        # 모델 목록 추출 로직 (유연하게 대응)
        if hasattr(model_list, 'models'):
            actual_names = [m.model for m in model_list.models]
        elif isinstance(model_list, dict) and 'models' in model_list:
            actual_names = [m['name'] for m in model_list['models']]
        else:
            # 다른 형태일 경우 문자열 변환 후 파싱 시도
            actual_names = [str(model_list)]

        print(f"📋 라이브러리 인식 목록: {actual_names}")
        
        target = 'MedAIBase/Qwen3-VL-Reranker:2b'
        # 가장 유사한 이름 찾기 (부분 일치 지원)
        matched_name = None
        for name in actual_names:
            if target.lower() in name.lower() or name.lower() in target.lower():
                matched_name = name
                break
        
        if not matched_name:
            print(f"❌ 목록에서 '{target}' 유사 모델을 못 찾았습니다.")
            return

        print(f"🎯 매칭된 모델명: '{matched_name}'")
        
        query = "France Capital?"
        doc = "Paris."
        prompt = f"Query: {query}\nDoc: {doc}\nScore:"

        print(f"⏳ 추론 시작...")
        start_time = time.time()
        
        response = ollama.generate(
            model=matched_name,
            prompt=prompt
        )
        
        print(f"✅ 성공! ({time.time()-start_time:.2f}s)")
        print(f"📥 결과: {response['response']}")

    except Exception as e:
        print(f"❌ 실패: {e}")

if __name__ == "__main__":
    diagnose_and_retry()