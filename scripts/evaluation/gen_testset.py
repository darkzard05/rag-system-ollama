"""
로컬 LLM에 최적화된 경량 RAG 테스트셋 생성기.
PDF 핵심 내용을 추출하여 질문-정답-컨텍스트 쌍을 생성합니다.
"""

import asyncio
import logging
import os
import pandas as pd
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 질문 생성을 위한 프롬프트
GENERATE_PROMPT = """
당신은 교육 전문가입니다. 제공된 [문서 내용]을 바탕으로 질문과 답변 세트를 만드세요.

[문서 내용]:
{content}

[규칙]:
1. 반드시 문서에 명시된 사실만을 바탕으로 질문하세요.
2. 답변은 문서의 내용을 요약하거나 인용하여 정확하게 작성하세요.
3. 결과는 반드시 아래 JSON 형식을 따르세요.
{{
    "qa_pairs": [
        {{
            "question": "질문 1",
            "answer": "정답 1"
        }},
        {{
            "question": "질문 2",
            "answer": "정답 2"
        }}
    ]
}}
"""

async def generate_simple_testset(file_path: str, test_size: int = 5):
    """
    로컬 LLM을 사용하여 PDF에서 질문-답변 세트를 추출합니다.
    """
    logger.info(f"데이터셋 생성 시작: {file_path}")
    
    # 1. 문서 로드 (앞부분 5페이지 정도만 샘플링)
    loader = PyMuPDFLoader(file_path)
    all_docs = loader.load()
    docs = all_docs[:10] # 전체가 아닌 일부 페이지만 사용
    
    llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    
    all_qa_pairs = []
    
    # 페이지별로 질문 생성
    for i, doc in enumerate(docs):
        if len(all_qa_pairs) >= test_size:
            break
            
        if len(doc.page_content.strip()) < 200:
            continue
            
        try:
            logger.info(f"페이지 {i+1} 분석 중...")
            result = await chain.ainvoke({"content": doc.page_content[:2000]})
            
            for pair in result.get("qa_pairs", []):
                all_qa_pairs.append({
                    "question": pair["question"],
                    "ground_truth": pair["answer"],
                    "reference_context": doc.page_content
                })
                if len(all_qa_pairs) >= test_size:
                    break
        except Exception as e:
            logger.warning(f"페이지 {i+1} 처리 실패: {e}")
            continue
            
    # 2. 결과 저장
    df = pd.DataFrame(all_qa_pairs)
    output_dir = "tests/data"
    os.makedirs(output_dir, exist_ok=True)
    
    file_base = os.path.basename(file_path).split('.')[0]
    output_path = f"{output_dir}/testset_{file_base}.csv"
    
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"데이터셋 생성 완료 ({len(all_qa_pairs)}개): {output_path}")
    return output_path

if __name__ == "__main__":
    target_pdf = "tests/data/2201.07520v1.pdf"
    if os.path.exists(target_pdf):
        asyncio.run(generate_simple_testset(target_pdf, test_size=10))
    else:
        logger.error(f"대상 파일이 없습니다: {target_pdf}")