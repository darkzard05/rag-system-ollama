import time
import logging
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any

# 프로젝트 루트를 경로에 추가


from langchain_core.documents import Document
from src.core.reranker import BGEReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    name: str
    query: str
    docs: List[Document]
    positive_idx: int  # 정답 문서의 인덱스


def run_reranker_test():
    reranker = BGEReranker()
    if not reranker.model:
        print("❌ 리랭커 초기화 실패. 테스트를 중단합니다.")
        return

    # 테스트 케이스 정의
    test_cases = [
        TestCase(
            name="Obvious Match (키워드 일치)",
            query="아이폰 15의 배터리 용량은 얼마인가요?",
            docs=[
                Document(
                    page_content="아이폰 15의 배터리 용량은 3,349mAh입니다."
                ),  # Positive
                Document(
                    page_content="삼성 갤럭시 S23의 배터리 성능은 매우 뛰어납니다."
                ),
                Document(
                    page_content="최근 스마트폰 시장에서는 접이식 폰이 유행하고 있습니다."
                ),
                Document(page_content="배터리 관리 팁: 화면 밝기를 낮추세요."),
            ],
            positive_idx=0,
        ),
        TestCase(
            name="Semantic Match (의미론적 일치)",
            query="이 회사의 수익 구조에 대해 설명해줘",
            docs=[
                Document(
                    page_content="지난 분기 매출액은 1조 원이며, 영업이익률은 15%를 기록했습니다."
                ),  # Positive
                Document(page_content="회사는 서울 강남구에 본사를 두고 있습니다."),
                Document(
                    page_content="임직원 복지 제도에는 유연 근무제가 포함되어 있습니다."
                ),
                Document(page_content="회사의 설립 연도는 2010년입니다."),
            ],
            positive_idx=0,
        ),
        TestCase(
            name="Hard Negative (키워드는 겹치나 무관)",
            query="딥러닝 모델의 최적화 방법",
            docs=[
                Document(
                    page_content="최적화된 운송 경로를 통해 물류 비용을 절감했습니다."
                ),  # Negative (최적화 키워드만 겹침)
                Document(
                    page_content="Adam 옵티마이저를 사용하여 학습률을 동적으로 조절하고 손실 함수를 최소화합니다."
                ),  # Positive
                Document(page_content="딥러닝의 역사에 대해 알아보겠습니다."),
                Document(
                    page_content="모델의 파라미터 수가 많아질수록 연산량이 증가합니다."
                ),
            ],
            positive_idx=1,
        ),
        TestCase(
            name="No Match (관련 정보 없음)",
            query="내일 날씨가 어때?",
            docs=[
                Document(page_content="이 문서는 기업의 재무제표 분석 보고서입니다."),
                Document(page_content="반도체 공정의 미세화 기술에 대한 설명입니다."),
                Document(
                    page_content="인공지능의 윤리적 가이드라인에 관한 내용입니다."
                ),
                Document(page_content="파이썬 프로그래밍 언어의 기초 문법입니다."),
            ],
            positive_idx=-1,  # 정답 없음
        ),
    ]

    print("\n" + "=" * 80)
    print(
        f"{'Test Case':<30} | {'Pos Rank':<10} | {'Pos Score':<10} | {'Time(ms)':<10} | {'Result'}"
    )
    print("-" * 80)

    total_latency = 0
    hits = 0

    for tc in test_cases:
        start_time = time.perf_counter()
        # 리랭킹 실행 (top_k는 전체 문서 수로 설정하여 순위 확인)
        reranked_docs = reranker.rerank_documents(tc.query, tc.docs, top_k=len(tc.docs))
        latency = (time.perf_counter() - start_time) * 1000
        total_latency += latency

        # 정답 문서 찾기
        pos_rank = -1
        pos_score = 0.0

        if tc.positive_idx != -1:
            # 원본 문서의 내용을 기준으로 정답 문서의 순위를 찾음
            pos_content = tc.docs[tc.positive_idx].page_content
            for i, doc in enumerate(reranked_docs):
                if doc.page_content == pos_content:
                    pos_rank = i + 1
                    pos_score = doc.metadata.get("rerank_score", 0.0)
                    break

            result = "✅ PASS" if pos_rank == 1 else "❌ FAIL"
            if pos_rank == 1:
                hits += 1
        else:
            # 정답이 없는 경우, 상위 문서의 점수가 낮아야 함
            top_score = (
                reranked_docs[0].metadata.get("rerank_score", 0.0)
                if reranked_docs
                else 0.0
            )
            pos_rank = "N/A"
            pos_score = top_score
            result = "✅ PASS" if top_score < 0.5 else "❌ FAIL"  # 임계값 0.5 가정

        print(
            f"{tc.name:<30} | {str(pos_rank):<10} | {pos_score:<10.4f} | {latency:<10.2f} | {result}"
        )

    print("-" * 80)
    print(f"평균 지연 시간: {total_latency / len(test_cases):.2f}ms")
    print(
        f"Hit Rate@1: {hits / (len(test_cases) - 1) * 100:.1f}% (excluding No Match case)"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_reranker_test()
