"""FlashRank 리랭커 벤치마크 스크립트 (P3 기준치 수집용).

2201.07520v1.pdf 실제 검색 청크(벡터 캐시 로드 실패 시 합성 한국어 섹션)
18개 후보 풀에 대해 질의 "cm3가 뭔가요?"의 리랭크 지연시간을 반복 측정하고
reports/rerank_bench_<ts>.json 으로 기록한다.

사용법:
    python scripts/benchmark_reranker.py --repeat 5 [--model <name>]

온디맨드 실행이므로 각 rerank 호출에 ~8초(CPU)가 소요된다. 중간에 종료하지
말 것. stdout 전체를 보고용 파일로 캡처해 두는 것이 좋다.
"""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

QUERY = "cm3가 뭔가요?"
POOL_SIZE = 18
PDF_PATH = ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf"
MODEL_CACHE_DIR = str(ROOT_DIR / ".model_cache")


class _StubEmbedder:
    """VectorStoreCache.load()가 FAISS 래퍼 구성에만 사용하는 임베더 (추론 없음)."""

    def embed_query(self, text: str) -> list[float]:
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


def _synthetic_sections() -> tuple[str, ...]:
    """실제 캐시 로드가 불가할 때 쓰는 합성 한국어 섹션 18개 (~1000자 + 초단문 1개)."""
    return (
        "## CM3 개요\n\nCM3는 Meta AI가 발표한 인과적 마스킹 멀티모달 모델(Causal Masked Multimodal Model)로, "
        "이미지와 텍스트를 하나의 공통 토큰 공간에서 처리하는 통합 생성 모델이다. 기존의 CLIP 방식처럼 이미지와 "
        "텍스트를 별도의 임베딩 공간으로 보내 대조 학습(contrastive learning)을 수행하는 대신, CM3는 두 양식을 모두 "
        "이산(discrete) 토큰으로 변환한 뒤 하나의 시퀀스로 이어 붙여 Transformer에 입력한다. 이렇게 하면 모델이 "
        "이미지 캡셔닝, 텍스트 기반 이미지 생성, 이미지 편집, 시각적 질의응답 등 여러 작업을 별도의 파인튜닝 없이도 "
        "같은 파라미터로 수행할 수 있다. 언어 모델링에서 검증된 아키텍처와 학습 기법을 이미지 도메인으로 그대로 "
        "확장한 대표 사례로, 생성 품질과 데이터 효율성 모두 기존 접근법을 능가하는 결과를 보여주었다. 특히 대조 학습이 "
        "요구하는 대규모 배치와 음수 샘플링 없이도 단일 모델이 다양한 멀티모달 작업을 처리할 수 있다는 점에서, "
        "이후 등장한 생성 모델들의 아키텍처 설계에 큰 영향을 주었다.",
        "## 토크나이저와 특수 토큰\n\nCM3는 텍스트와 이미지 토큰을 함께 다루기 위해 SentencePiece 기반 토크나이저를 "
        "확장하여 사용한다. 이미지 시퀀스의 시작과 끝을 나타내는 BOI(Beginning Of Image), EOI(End Of Image)와 "
        "텍스트 시퀀스의 시작과 끝을 나타내는 BOT(Beginning Of Text), EOT(End Of Text) 같은 특수 토큰을 도입해 "
        "양식 간 경계를 명시적으로 표현한다. 이 특수 토큰 덕분에 모델은 이미지와 텍스트가 섞인 혼합 시퀀스를 "
        "문맥에 따라 유연하게 생성할 수 있다. 예를 들어 텍스트 설명이 먼저 오고 이미지 토큰이 이어지는 패턴이나, "
        "이미지 토큰 다음에 캡션이 오는 패턴 모두 자연스럽게 학습된다. 토크나이저의 어휘 크기와 서브워드 분할 방식은 "
        "텍스트의 언어 특성에 맞게 조정되며, 이미지 토큰은 별도의 코드북 인덱스로 표현되어 텍스트 어휘와 충돌하지 "
        "않도록 설계되었다. 이처럼 명시적인 경계 토큰을 사용하면 복잡한 멀티모달 작업을 하나의 자기회귀 시퀀스 "
        "생성 문제로 단순화할 수 있다는 장점이 있다.",
        "## 모델 구조와 학습 목적\n\nCM3의 핵심 구조는 causal masked Transformer로, 표준 자기회귀 언어 모델의 "
        "구조를 유지하면서 학습 시 일부 토큰을 마스킹하여 예측하도록 훈련한다. 순수한 인과적 마스킹(causal masking) "
        "조건에서 학습하므로 추론 시에는 완전한 자기회귀 디코딩을 그대로 사용할 수 있고, 학습 단계에서는 일부 위치의 "
        "토큰을 가려 그 토큰의 원래 값을 예측하는 방식으로 손실을 계산한다. 이를 통해 문장 생성과 같은 순방향 생성 "
        "작업뿐 아니라 이미지 편집처럼 입력과 출력 사이에 조건부 관계가 있는 작업도 같은 모델로 처리할 수 있다. "
        "학습 목적 함수는 기본적으로 토큰 단위 교차 엔트로피 손실을 사용하며, 마스킹된 토큰의 비율과 위치는 학습 "
        "과정에서 무작위로 결정된다. 이 구조는 이후 DALL-E와 같은 모델이 순수 자기회귀 학습만으로 이미지를 생성하는 "
        "방식과 차별화되는 지점이며, 조건부 생성과 무조건부 생성을 하나의 파라미터 집합으로 통합할 수 있게 한다.",
        "## 이미지 토큰화 (이산 VAE)\n\n이미지를 이산 토큰으로 변환하기 위해 CM3는 이산 VAE(discrete VAE)를 "
        "사용한다. 이미지를 고정 크기의 격자로 나눈 뒤 각 영역을 학습된 코드북의 인덱스로 매핑하여, 이미지 전체를 "
        "일련의 정수 토큰으로 압축한다. 이 과정에서 이미지의 저수준 픽셀 정보는 VAE의 인코더-디코더 구조를 통해 "
        "재구성 가능한 형태로 보존된다. 학습된 코드북의 크기와 각 이미지가 분할되는 토큰 수는 모델의 표현력과 "
        "계산 비용 사이의 트레이드오프를 결정하는 핵심 하이퍼파라미터다. 코드북이 너무 작으면 미세한 시각적 디테일이 "
        "손실되고, 코드북이 너무 크면 학습이 어려워진다. 이산 VAE를 사용하는 이유는 텍스트와 이미지가 동일한 정수 "
        "토큰 공간에 놓이게 되어 Transformer가 두 양식을 단일 시퀀스로 처리할 수 있기 때문이다. 이러한 접근은 "
        "이미지 생성에서 널리 쓰이는 방식으로, 재구성 품질과 생성 품질 모두에서 우수한 성능을 보여준다.",
        "## 학습 데이터와 사전학습\n\nCM3는 대규모 웹 데이터에서 수집한 이미지-텍스트 쌍으로 사전학습을 수행한다. "
        "수집된 데이터는 중복 제거, 필터링, 이미지 품질 검사 등의 정제 과정을 거치며, 각 이미지에 대한 대체 텍스트와 "
        "메타데이터가 함께 저장된다. 이렇게 정제된 데이터셋에서 모델은 이미지와 텍스트가 짝을 이루는 패턴을 학습하게 "
        "되고, 이를 통해 이미지와 텍스트 사이의 의미적 연관성을 파악한다. 사전학습 데이터의 규모와 다양성은 모델의 "
        "성능을 결정하는 가장 중요한 요소 중 하나이며, CM3는 수억 개 이상의 이미지-텍스트 쌍을 사용한 것으로 "
        "알려져 있다. 또한 학습 데이터에 이미지 편집 전후 쌍을 포함시켜 편집 작업에 필요한 지식을 함께 학습하며, "
        "이 덕분에 별도의 편집 데이터셋을 모으지 않아도 지시 기반 편집이 가능해진다. 데이터 효율성 측면에서 CM3는 "
        "동일한 데이터 규모를 사용한 기존 모델 대비 더 나은 성능을 보여준 것으로 보고되었다.",
        "## 생성 과정과 하이브리드 디코딩\n\nCM3는 추론 시 여러 가지 디코딩 전략을 결합한 하이브리드 방식을 "
        "사용한다. 기본적으로 자기회귀 디코딩을 통해 토큰을 순차적으로 생성하되, 모델이 스스로 자신의 예측에 대한 "
        "확신도를 평가하여 상황에 따라 다른 생성 방식을 선택한다. 예를 들어 모델이 입력 문맥에 대해 높은 확신을 "
        "가지고 있을 때는 빔 서치 같은 결정적 방법을 사용해 안정적인 출력을 만들고, 불확실성이 높은 영역에서는 "
        "샘플링을 통해 다양한 후보를 탐색한다. 이 과정에서 모델의 내부 활성화 값이나 토큰 확률 분포로부터 불확실성을 "
        "추정하며, 이를 기반으로 어떤 영역에서 어떤 전략을 쓸지 결정한다. 이러한 하이브리드 접근은 순수 자기회귀 "
        "디코딩보다 더 나은 품질과 더 낮은 계산 비용을 동시에 달성할 수 있게 한다. 특히 긴 이미지 시퀀스를 생성할 때 "
        "모든 토큰을 순차적으로 생성하는 대신 병렬로 생성할 수 있는 구간을 찾아내어 추론 속도를 크게 개선한다.",
        "## 이미지 편집과 조건부 생성\n\nCM3의 중요한 응용 중 하나는 이미지 편집이다. 모델이 이미지와 텍스트를 "
        "같은 토큰 공간에서 다루기 때문에, 편집할 이미지를 입력으로 주고 '이 이미지의 배경을 밤하늘로 바꿔줘' 같은 "
        "텍스트 지시를 덧붙이면 모델이 조건에 맞는 새로운 이미지 토큰 시퀀스를 생성한다. 이 과정에서 모델은 입력 "
        "이미지의 구조적 정보를 보존하면서도 텍스트로 표현된 변화만 적용하는 법을 학습한다. 단순한 스타일 변경부터 "
        "객체 추가, 배경 교체, 색상 변경 등 다양한 편집이 가능하며, 편집의 대상이 되는 영역을 텍스트로 명시할 수 "
        "있다. 또한 마스킹 기반 학습 덕분에 이미지의 한 부분만 가린 뒤 그 부분을 채워 넣는 인페인팅 작업도 가능하다. "
        "이러한 조건부 생성 능력은 양식 간 토큰을 하나의 시퀀스로 통합한 설계 덕분에 추가적인 모듈 없이 자연스럽게 "
        "구현되며, 사용자에게 직관적인 인터페이스를 제공한다.",
        "## 제로샷 생성과 데이터 효율성\n\nCM3는 사전학습만으로 다양한 작업에 대한 제로샷(zero-shot) 성능을 "
        "보여준다. 텍스트로부터 새로운 이미지를 생성하는 텍스트-이미지 생성, 이미지를 설명하는 캡션 생성, 그리고 "
        "이미지 편집 작업 모두 추가적인 학습 없이 수행할 수 있다. 특히 데이터 효율성 측면에서 CM3는 기존 생성 모델과 "
        "비교했을 때 동일한 성능을 달성하기 위해 필요한 데이터가 절반 수준에 불과하다는 결과를 보고했다. 이는 "
        "대조 학습 방식과 달리 정보가 풍부한 생성 목적 함수를 사용하기 때문으로 분석된다. 생성 목적 함수는 모델이 "
        "데이터의 분포 자체를 학습하도록 만들어 주기 때문에, 제한된 데이터에서도 일반화 능력을 효과적으로 향상시킬 "
        "수 있다. 제로샷 능력은 모델 규모가 커질수록 더 두드러지며, 이는 언어 모델에서 관찰된 확장 법칙이 멀티모달 "
        "도메인에서도 유효함을 시사한다.",
        "## 기존 DALL-E와의 비교\n\nCM3는 같은 시기에 발표된 DALL-E와 자주 비교된다. DALL-E가 순수 자기회귀 "
        "모델로 텍스트 조건을 바탕으로 이미지를 생성하는 데 집중한 반면, CM3는 인과적 마스킹 학습을 통해 생성과 "
        "추론을 모두 지원한다는 차이가 있다. DALL-E는 학습 과정에서 텍스트에서 이미지로의 단방향 매핑만 학습하지만, "
        "CM3는 이미지에서 텍스트로, 텍스트에서 이미지로의 양방향 관계를 마스킹 학습으로 동시에 학습한다. 따라서 "
        "CM3는 캡셔닝, 편집, 조건부 생성과 같은 다양한 작업을 단일 모델로 처리할 수 있지만 DALL-E는 주로 생성 "
        "작업에 특화되어 있다. 또한 생성 품질의 경우 CM3가 인간 평가와 자동 지표 모두에서 DALL-E를 능가하는 "
        "결과를 보여주었으며, 특히 긴 텍스트 설명을 바탕으로 한 복잡한 이미지 생성에서 두드러진 차이를 보였다.",
        "## 평가 지표와 실험 결과\n\nCM3는 생성 품질을 여러 자동 지표와 인간 평가로 측정했다. 텍스트-이미지 "
        "생성의 경우 이미지-텍스트 정렬도를 측정하는 CLIP 스코어와 FID 스코어를 사용하며, 생성된 이미지가 텍스트 "
        "설명과 얼마나 일치하는지를 정량적으로 평가한다. 이미지 편집의 경우 원본 이미지 대비 편집 품질을 측정하는 "
        "별도의 지표를 사용하고, 캡셔닝은 표준 텍스트 생성 지표인 BLEU와 CIDEr로 평가한다. 실험 결과 CM3는 "
        "텍스트-이미지 생성에서 기존 방법보다 높은 CLIP 스코어를 기록했고, 인간 평가에서도 더 선호되는 결과를 "
        "만들어 냈다. 또한 데이터를 절반만 사용한 학습에서도 완전한 데이터로 학습한 기존 모델과 동등하거나 더 나은 "
        "성능을 보여주어 데이터 효율성의 우위를 입증했다. 이러한 결과는 생성 목적 함수 기반 학습이 멀티모달 "
        "도메인에서 강력한 선택지임을 실증적으로 뒷받침한다.",
        "## 확장성과 계산 비용\n\nCM3 아키텍처는 모델 파라미터 수와 학습 데이터 규모를 함께 확장할 수 있도록 "
        "설계되었다. Transformer 기반 구조는 언어 모델에서 이미 검증된 확장 법칙을 따르며, 모델 크기를 키울수록 "
        "이미지와 텍스트 생성 품질이 함께 향상된다. 다만 모델이 커질수록 학습과 추론에 필요한 계산 자원도 선형 "
        "이상으로 증가하는데, 특히 이미지 토큰 시퀀스는 텍스트보다 훨씬 길어 주의 계산(attention) 비용이 크게 "
        "늘어난다. 이를 완화하기 위해 CM3는 하이브리드 디코딩과 병렬 생성 전략을 사용해 추론 비용을 절감한다. "
        "또한 학습 단계에서는 대규모 병렬 처리와 그라디언트 체크포인팅 같은 기법을 적용해 메모리 사용을 최적화한다. "
        "이처럼 확장성과 효율성을 함께 고려한 설계 덕분에 CM3는 실제 서비스 규모에서도 활용 가능한 수준의 추론 "
        "속도를 달성할 수 있었다.",
        "## 추론 시 불확실성 활용\n\nCM3의 하이브리드 디코딩 전략의 핵심은 모델이 스스로 예측의 확신도를 "
        "평가하는 것이다. 모델은 각 토큰 위치에서 예측 확률 분포의 엔트로피나 여러 예측 간의 합의 정도를 계산하여 "
        "불확실성을 수치화한다. 확신도가 높은 영역에서는 빔 서치나 그리디 디코딩으로 결정적인 출력을 만들고, "
        "불확실성이 높은 영역에서는 온도가 조절된 샘플링으로 더 다양한 후보를 탐색한다. 이 과정에서 생성된 중간 "
        "후보들에 대한 추가적인 스코어링을 수행하여 최종 출력을 선택하는데, 이는 검색 기반 생성과 확률적 생성을 "
        "절충한 방식이다. 이러한 접근은 고정된 디코딩 전략을 사용할 때 발생하는 품질 저하와 비효율을 줄여준다. "
        "특히 이미지 생성처럼 출력 공간이 매우 넓은 작업에서 불확실성 기반 전략 선택은 탐색과 활용 사이의 균형을 "
        "효과적으로 맞춘다.",
        "## 다국어 및 도메인 확장\n\nCM3의 토크나이저는 다국어 텍스트를 지원하도록 설계되어 영어뿐 아니라 "
        "다양한 언어의 텍스트 조건으로 이미지를 생성할 수 있다. SentencePiece 토크나이저는 언어에 구애받지 않는 "
        "서브워드 분할을 제공하므로, 학습 데이터에 포함된 모든 언어의 텍스트를 동일한 어휘 공간에서 처리할 수 있다. "
        "이를 통해 한국어, 일본어, 중국어 같은 비영어권 언어로도 자연스러운 이미지 생성이 가능하며, 기술 문서나 "
        "과학 논문과 같은 전문 도메인의 텍스트에도 적용할 수 있다. 도메인 확장의 경우 의료 영상, 위성 사진, 패션 "
        "이미지 등 특수한 이미지 분포에 대한 미세 조정을 통해 전문 분야에서도 우수한 성능을 낼 수 있다. 이처럼 "
        "언어와 도메인의 유연성은 CM3가 일반 목적의 멀티모달 시스템으로 활용될 수 있는 근거가 된다.",
        "## 파인튜닝과 지시 학습\n\n사전학습된 CM3는 특정 작업이나 도메인에 맞게 파인튜닝할 수 있다. 파인튜닝은 "
        "작은 규모의 작업별 데이터셋으로 모델 전체 또는 일부 계층을 업데이트하는 방식으로 이루어지며, 지시 "
        "튜닝(instruction tuning)을 통해 사용자의 자연어 지시를 더 정확하게 따르도록 만들 수도 있다. 지시 튜닝 "
        "데이터에는 텍스트-이미지 생성 요청, 이미지 편집 지시, 캡션 생성 요청 등 다양한 형태의 명령어가 포함되며, "
        "모델은 이러한 예시를 통해 명령의 의도를 파악하고 적절한 출력 형식으로 응답하는 법을 학습한다. 파인튜닝 "
        "과정에서 생성 품질을 유지하기 위해 원본 사전학습 데이터의 일부를 함께 사용하는 경우도 많다. 또한 RLHF와 "
        "같은 인간 피드백 기반 정렬 기법을 적용하면 모델의 출력이 인간의 선호도에 더 잘 부합하게 되어 실제 서비스에 "
        "배포할 때 안전성과 유용성을 확보할 수 있다.",
        "## 한계점\n\nCM3 또한 여러 한계를 가지고 있다. 첫째, 이산 VAE로 이미지를 토큰화하는 과정에서 고주파 "
        "디테일과 미세한 질감 정보가 손실될 수 있다. 둘째, 이미지 토큰 시퀀스가 길어질수록 추론 비용이 급격히 "
        "증가하여 고해상도 이미지 생성에 많은 계산 자원이 필요하다. 셋째, 학습 데이터에 포함된 편향이 그대로 "
        "반영되어 특정 인종, 성별, 문화권에 대한 편향된 결과를 생성할 위험이 있다. 넷째, 어휘가 많은 언어나 "
        "희귀한 도메인 용어에 대해서는 토크나이저의 분할 품질이 저하될 수 있다. 다섯째, 파인튜닝 없이 정확한 "
        "기하학적 구조가 필요한 복잡한 장면을 생성하는 데 어려움이 있으며, 객체 간의 관계나 공간적 배열을 정확히 "
        "표현하지 못하는 경우가 많다. 이러한 한계는 이후 모델들이 개선해야 할 주요 과제로 남아 있다.",
        "## 향후 방향\n\nCM3의 설계는 이후 멀티모달 생성 연구의 방향을 제시하였다. 이미지와 텍스트를 단일 토큰 "
        "공간으로 통합하는 접근은 이후 더 큰 규모의 모델에서도 유지되었으며, 토큰화 방식과 디코딩 전략은 지속적으로 "
        "개선되고 있다. 특히 하이브리드 디코딩에서 사용된 불확실성 기반 생성 전략은 추론 효율성을 높이는 중요한 "
        "아이디어로 자리 잡았다. 학습 목적 함수 측면에서도 마스킹과 인과적 예측의 혼합 방식은 다양한 신경망 "
        "구조로 확장되었다. 또한 인간 피드백 기반 정렬, 안전성 필터, 콘텐츠 검증 시스템을 결합하는 연구가 "
        "진행되고 있어 모델의 실용성과 신뢰성이 함께 향상될 것으로 기대된다. 궁극적으로는 텍스트, 이미지, 오디오, "
        "비디오를 모두 포함하는 완전한 멀티모달 기반 모델로의 확장이 목표가 되고 있다.",
        "## 요약 및 의의\n\nCM3는 이미지와 텍스트를 하나의 토큰 시퀀스로 통합하고 인과적 마스킹 학습을 적용한 "
        "대표적인 멀티모달 생성 모델이다. 대조 학습 없이 생성 목적 함수만으로 학습하면서도 여러 작업에서 제로샷 "
        "성능을 보여주었고, 데이터 효율성과 생성 품질 모두 기존 방법을 능가하는 결과를 냈다. 하이브리드 디코딩과 "
        "불확실성 기반 전략 선택은 추론 효율성과 품질 사이의 균형을 잡는 새로운 패러다임을 제시했다. 비록 이산 "
        "토큰화로 인한 디테일 손실과 고해상도 생성의 계산 비용 같은 한계가 남아 있지만, CM3의 통합 토큰 공간 "
        "설계와 생성 기반 학습 방식은 이후 멀티모달 연구의 기반이 되었다. 요약하면 CM3는 텍스트와 이미지를 "
        "동등하게 다루는 통합 생성 모델의 가능성을 실증한 중요한 작업이다.",
        "## 참고\n\n이 섹션은 짧은 내용입니다.",
    )


def _build_synthetic_pool() -> list[Any]:
    from langchain_core.documents import Document

    return [Document(page_content=section) for section in _synthetic_sections()]


def _embedding_candidates() -> list[str]:
    try:
        from common.config import DEFAULT_EMBEDDING_MODEL

        default = DEFAULT_EMBEDDING_MODEL
    except Exception:
        default = "nomic-embed-text-v2-moe"
    candidates: list[str] = []
    for name in (default, "nomic-embed-text-v2-moe", "nomic-embed-text"):
        if name not in candidates:
            candidates.append(name)
    return candidates


def _load_real_pool() -> tuple[list[Any] | None, list[str]]:
    """벡터 캐시에서 2201.07520v1.pdf 실제 검색 청크 18개를 로드한다.

    실패 시 (None, errors) 반환 — 호출부에서 합성 풀로 폴백한다.
    """
    errors: list[str] = []
    if not PDF_PATH.is_file():
        errors.append(f"pdf missing: {PDF_PATH}")
        return None, errors
    try:
        from cache.vector_cache import VectorStoreCache

        for embedding_model in _embedding_candidates():
            cache = VectorStoreCache(str(PDF_PATH), embedding_model)
            if not Path(cache.cache_dir).is_dir():
                continue
            splits, _, _ = cache.load(_StubEmbedder())
            if splits and len(splits) >= POOL_SIZE:
                return splits[:POOL_SIZE], errors
            errors.append(
                f"cache load ({embedding_model}) returned "
                f"{0 if splits is None else len(splits)} docs"
            )
    except Exception as e:
        errors.append(f"vector cache load failed: {type(e).__name__}: {e}")
    return None, errors


def _build_pool() -> tuple[list[Any], str, list[str]]:
    pool, errors = _load_real_pool()
    if pool is not None:
        return pool, "real", errors
    return _build_synthetic_pool(), "synthetic", errors


def _probe_providers() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers()), errors
    except Exception as e:
        errors.append(f"onnxruntime unavailable: {type(e).__name__}: {e}")
        return [], errors


def _build_ranker(model_name: str) -> tuple[Any | None, list[str]]:
    errors: list[str] = []
    try:
        from flashrank import Ranker

        # [stale-state 가드] 스크립트 프로세스 내에서 항상 신규 Ranker를 구성한다
        # (전역 워밍 객체 재사용 금지 — 측정 대상은 콜드 세션 생성 포함).
        return Ranker(model_name=model_name, cache_dir=MODEL_CACHE_DIR), errors
    except Exception as e:
        errors.append(f"flashrank Ranker load failed: {type(e).__name__}: {e}")
        return None, errors


def _run_rerank_loop(
    ranker: Any, pool: list[Any], query: str, repeat: int
) -> tuple[list[float], list[str]]:
    from flashrank import RerankRequest

    errors: list[str] = []
    per_run_ms: list[float] = []
    for _ in range(repeat):
        request = RerankRequest(
            query=query,
            passages=[
                {"id": i, "text": doc.page_content} for i, doc in enumerate(pool)
            ],
        )
        start = time.perf_counter()
        try:
            ranker.rerank(request)
        except Exception as e:
            errors.append(f"rerank run failed: {type(e).__name__}: {e}")
            break
        per_run_ms.append((time.perf_counter() - start) * 1000.0)
    return per_run_ms, errors


def _collect_session_info(ranker: Any) -> dict[str, Any] | str:
    session = getattr(ranker, "session", None)
    if session is None:
        return "session internals not exposed"
    try:
        providers = session.get_providers()
    except Exception as e:
        return {"providers_error": f"{type(e).__name__}: {e}"}
    info: dict[str, Any] = {"providers": list(providers)}
    try:
        options = session.get_session_options()
        info["intra_op_num_threads"] = options.intra_op_num_threads
        info["inter_op_num_threads"] = options.inter_op_num_threads
        info["execution_mode"] = str(options.execution_mode)
        info["graph_optimization_level"] = str(options.graph_optimization_level)
    except Exception as e:
        info["options_error"] = f"{type(e).__name__}: {e}"
    return info


def _p95_ms(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _write_report(report: dict[str, Any]) -> Path:
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"rerank_bench_{report['timestamp']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_summary(report: dict[str, Any], out_path: Path) -> None:
    print(f"[bench] model={report['model']}")
    print(
        f"[bench] pool_type={report['pool_type']} "
        f"num_candidates={report['num_candidates']}"
    )
    print(f"[bench] query={report['query']}")
    print(f"[bench] onnxruntime providers={list(report['providers'])}")
    session_info = report["session_info"]
    if isinstance(session_info, dict):
        print(
            f"[bench] session providers={list(session_info['providers'])} "
            f"intra_op_num_threads={session_info.get('intra_op_num_threads')} "
            f"inter_op_num_threads={session_info.get('inter_op_num_threads')} "
            f"graph_optimization_level={session_info.get('graph_optimization_level')}"
        )
    else:
        print(f"[bench] {session_info}")
    print(f"[bench] per_run_ms={report['per_run_ms']}")
    print(
        f"[bench] mean_ms={report['mean_ms']} median_ms={report['median_ms']} "
        f"p95_ms={report['p95_ms']}"
    )
    if report["errors"]:
        print(f"[bench] errors={report['errors']}")
    print(f"[bench] report written to {out_path}")


def _run_benchmark(model_name: str, repeat: int) -> dict[str, Any]:
    providers, provider_errors = _probe_providers()
    pool, pool_type, pool_errors = _build_pool()

    ranker, ranker_errors = _build_ranker(model_name)
    errors = provider_errors + pool_errors + ranker_errors

    per_run_ms: list[float] = []
    if ranker is not None:
        per_run_ms, run_errors = _run_rerank_loop(ranker, pool, QUERY, repeat)
        errors.extend(run_errors)

    session_info: dict[str, Any] | str = (
        _collect_session_info(ranker)
        if ranker is not None
        else "session internals not exposed"
    )

    return {
        "model": model_name,
        "pool_type": pool_type,
        "num_candidates": len(pool),
        "query": QUERY,
        "repeat": repeat,
        "per_run_ms": [round(v, 2) for v in per_run_ms],
        "mean_ms": round(statistics.fmean(per_run_ms), 2) if per_run_ms else 0.0,
        "median_ms": round(statistics.median(per_run_ms), 2) if per_run_ms else 0.0,
        "p95_ms": round(_p95_ms(per_run_ms), 2) if per_run_ms else 0.0,
        "providers": providers,
        "session_info": session_info,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "errors": errors,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashRank 리랭커 벤치마크")
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="리랭크 반복 측정 횟수 (기본 5)",
    )
    parser.add_argument(
        "--model",
        default="ms-marco-MultiBERT-L-12",
        help="FlashRank 모델명 (기본: config.yml reranker.model_name)",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _run_benchmark(args.model, args.repeat)
    out_path = _write_report(report)
    _print_summary(report, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
