import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.model_loader import ModelManager
from langchain_core.runnables import RunnableBinding

def test_llm_caching_efficiency():
    print("=== LLM 인스턴스 캐싱 효율성 테스트 ===")
    
    # 모델명 (실제 로드 시도하지 않도록 모킹하거나 존재하는 이름 사용)
    # 여기서는 로직만 확인하기 위해 load_llm을 모킹함
    import core.model_loader
    from unittest.mock import MagicMock
    core.model_loader.load_llm = MagicMock(return_value=MagicMock())
    
    model_name = "test-model-123"
    
    # 1. 서로 다른 설정으로 여러 번 호출
    llm1 = ModelManager.get_llm(model_name, temperature=0.1)
    llm2 = ModelManager.get_llm(model_name, temperature=0.9)
    llm3 = ModelManager.get_llm(model_name, temperature=0.5, num_ctx=8192)
    
    # 2. 캐시된 베이스 인스턴스 확인
    base_instances = [k for k in ModelManager._instances.keys() if "base_llm_test-model-123" in k]
    print(f"생성된 베이스 인스턴스 키 리스트: {base_instances}")
    
    # 3. 결과 분석
    print(f"llm1 타입: {type(llm1)}")
    
    # RunnableBinding인지 확인 (bind()의 결과물)
    is_binding = isinstance(llm1, RunnableBinding)
    
    print(f"\n결과 요약:")
    print(f"1. 베이스 인스턴스 개수: {len(base_instances)}")
    print(f"2. 동적 바인딩 사용 여부: {is_binding}")
    
    if len(base_instances) == 1 and is_binding:
        print("\n✅ 성공: 메모리 효율적인 모델 관리 로직이 정상 작동합니다.")
    else:
        print("\n❌ 실패: 결과가 기대와 다릅니다.")

if __name__ == "__main__":
    test_llm_caching_efficiency()