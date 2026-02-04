import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.model_loader import ModelManager
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableBinding

def test_llm_caching_efficiency_real():
    print("=== LLM 인스턴스 캐싱 효율성 테스트 (실제 객체) ===")
    
    import core.model_loader
    from unittest.mock import MagicMock
    
    # 가짜 ChatOllama 객체 생성
    fake_llm = ChatOllama(model="test")
    core.model_loader.load_llm = MagicMock(return_value=fake_llm)
    
    model_name = "test-real-model-final"
    
    # 1. 서로 다른 설정으로 호출
    llm1 = ModelManager.get_llm(model_name, temperature=0.1)
    llm2 = ModelManager.get_llm(model_name, temperature=0.9)
    
    # 2. 결과 분석
    print(f"llm1 타입: {type(llm1)}")
    is_binding = isinstance(llm1, RunnableBinding)
    
    # 바인딩된 파라미터 확인
    bound_params = llm1.kwargs if is_binding else {}
    print(f"llm1 바인딩 파라미터: {bound_params}")
    
    base_count = sum(1 for k in ModelManager._instances.keys() if model_name in k)
    
    if base_count == 1 and is_binding and bound_params.get("temperature") == 0.1:
        print("결과: [성공] 단일 인스턴스 공유 및 동적 파라미터 바인딩 확인 완료.")
    else:
        print("결과: [실패] 로직에 결함이 있습니다.")

if __name__ == "__main__":
    test_llm_caching_efficiency_real()