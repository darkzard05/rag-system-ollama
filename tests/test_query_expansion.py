import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.query_optimizer import RAGQueryOptimizer
from src.core.graph_builder import build_graph
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage


class TestQueryExpansion:
    def test_is_complex_query_simple(self):
        """단순 질문 식별 테스트"""
        # 짧은 질문
        assert RAGQueryOptimizer.is_complex_query("안녕?") is False

        # 키워드 포함 ("제목")
        assert RAGQueryOptimizer.is_complex_query("이 문서의 제목이 뭐야?") is False

        # 단순 의문문 패턴 ("뭐야?")
        assert RAGQueryOptimizer.is_complex_query("이게 뭐야?") is False

    def test_is_complex_query_complex(self):
        """복잡한 질문 식별 테스트"""
        # 긴 질문 (40자 이상)
        long_query = "이 시스템에서 사용하는 RAG 파이프라인의 구성 요소와 각각의 역할에 대해 상세하게 설명해 주시겠습니까?"
        assert len(long_query) >= 40
        assert RAGQueryOptimizer.is_complex_query(long_query) is True

        # [변경] "장점" 키워드가 포함되어 있으므로 이제는 복잡한 질문(True)으로 판단해야 함
        mid_query = "RAG 시스템의 장점은 무엇인가요?"
        assert RAGQueryOptimizer.is_complex_query(mid_query) is True

        # [변경] "구성", "설명" 키워드가 포함되어 있으므로 True
        mid_query_analytic = "RAG 시스템 구성 요소 설명 부탁해"
        assert RAGQueryOptimizer.is_complex_query(mid_query_analytic) is True

        # [추가] 분석 키워드가 없고, 길이도 20자 미만이며, 의문사 패턴이 있는 경우 -> False
        # "언제야?" 패턴 포함, 길이 12자
        simple_query_with_pattern = "이 파일 언제 만들었어?"
        assert RAGQueryOptimizer.is_complex_query(simple_query_with_pattern) is False

    @pytest.mark.asyncio
    async def test_graph_query_expansion_logic(self):
        """그래프 내 쿼리 확장 노드 로직 테스트 (RunnableLambda 사용)"""

        # LLM 호출 감지용 Mock
        llm_call_tracker = AsyncMock()

        async def mock_llm_func(input_val):
            """Mock LLM 동작 함수"""
            # input_val은 PromptValue이거나 메시지 리스트임
            # 문자열로 변환하여 내용을 확인
            prompt_str = str(input_val)

            # 호출 기록
            await llm_call_tracker(prompt_str)

            # 쿼리 확장 프롬프트인지 확인 (간이 판별)
            # config.yml의 실제 내용: "Generate 3 search queries..."
            if (
                "3 search queries" in prompt_str
                or "similar to the user's question" in prompt_str
            ):
                return AIMessage(content="확장된 쿼리 1\n확장된 쿼리 2\n확장된 쿼리 3")
            else:
                # 일반 QA 답변
                return AIMessage(content="이것은 답변입니다.")

        # RunnableLambda로 래핑하여 파이프라인 호환성 확보
        mock_llm_runnable = RunnableLambda(mock_llm_func)

        # Retriever Mock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="test doc")]
        mock_retriever.ainvoke = AsyncMock(
            return_value=[Document(page_content="test doc")]
        )

        # Graph 빌드
        app = build_graph(retriever=mock_retriever)

        # --- Case 1: 단순 질문 ---
        simple_input = "제목이 뭐야?"
        config = {"configurable": {"llm": mock_llm_runnable}}

        # 트래커 리셋
        llm_call_tracker.reset_mock()
        mock_retriever.ainvoke.reset_mock()

        await app.ainvoke({"input": simple_input}, config=config)

        # 단순 질문 -> 쿼리 확장 안 함 -> 검색 -> 답변 생성
        expansion_calls_simple = [
            call
            for call in llm_call_tracker.call_args_list
            if "3 search queries" in str(call.args[0])
        ]
        assert len(expansion_calls_simple) == 0, (
            "단순 질문인데 쿼리 확장이 수행되었습니다."
        )

        # Retriever는 원본 질문으로 호출되어야 함
        retriever_calls_simple = [
            str(call.args[0]) for call in mock_retriever.ainvoke.call_args_list
        ]
        assert simple_input in retriever_calls_simple, (
            "원본 질문으로 검색하지 않았습니다."
        )

        # --- Case 2: 복잡한 질문 ---
        complex_input = (
            "이 시스템의 아키텍처와 성능 최적화 전략에 대해 자세히 설명해주고, 특히 캐싱 전략 이 어떻게 구성되어 있는지 알려줘."
            * 2
        )

        # 트래커 리셋
        llm_call_tracker.reset_mock()
        mock_retriever.ainvoke.reset_mock()

        await app.ainvoke({"input": complex_input}, config=config)

        # 복잡한 질문 -> 쿼리 확장 수행 -> 검색 -> 답변 생성
        expansion_calls_complex = [
            call
            for call in llm_call_tracker.call_args_list
            if "3 search queries" in str(call.args[0])  # config.yml 내용 반영
        ]

        assert len(expansion_calls_complex) > 0, (
            "복잡한 질문인데 쿼리 확장이 수행되지 않았습니다."
        )

        # Retriever 호출 확인
        retriever_calls_complex = [
            str(call.args[0]) for call in mock_retriever.ainvoke.call_args_list
        ]
        assert any("확장된 쿼리 1" in c for c in retriever_calls_complex)
