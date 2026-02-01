# Streaming Timeout Verification (Windows/Conda 포함)

이 문서는 **LLM 스트리밍이 타임아웃(기본 300초)으로 종료될 때**, 프로세스 로그에 아래와 같은 “잔여 async task / generator 종료 오류”가 발생하지 않는지 확인하기 위한 재현/검증 절차입니다.

## 목적
- 타임아웃 이후에도 앱이 정상 동작을 지속하는지 확인
- 로그에 아래 메시지들이 더 이상 반복되지 않는지 확인
  - `RuntimeError: async generator ignored GeneratorExit`
  - `sniffio._impl.AsyncLibraryNotFoundError: unknown async library, or not in async context`
  - `Task was destroyed but it is pending!`

## 사전 준비
- Ollama가 실행 중이어야 합니다.
- 앱 실행:

```bash
streamlit run src/main.py
```

## 재현 시나리오
1. PDF 업로드(이미 캐시가 있다면 로딩이 더 빠릅니다)
2. **일부러 답변이 길어지거나 느려질 질문**을 입력합니다.
   - 예시(긴 답변 유도):
     - “문서 전체를 섹션별로 요약하고, 각 섹션별 핵심 수식/정의도 설명해줘.”
     - “문서에서 등장하는 개념을 모두 뽑고, 각각의 정의/근거를 [p.X] 인용과 함께 정리해줘.”
3. 약 5분(300초) 내외로 타임아웃이 발생할 수 있습니다.

## 기대 결과(정상)
- UI에서 **부분 응답 또는 타임아웃 안내 메시지**가 출력됨
- 이후 추가 질문을 해도 앱이 계속 정상 응답
- 콘솔/로그에 위의 “ignored GeneratorExit / sniffio / pending task”가 **반복적으로 출력되지 않음**

## 문제가 남아있다면
- 해당 시점의 `logs/app.log`(또는 콘솔 로그) 일부를 공유해 주세요.
- 환경 정보:
  - `python -V`
  - `python -m pip show streamlit langchain langchain-core langgraph httpx httpcore sniffio`

