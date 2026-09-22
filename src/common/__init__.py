"""common 패키지 (Phase 1A 병합).

진원 모듈: ``utils`` (해시/캐시/주석), ``text_utils`` (텍스트/BM25),
``async_worker`` (비동기 워커), ``resilience`` (재시도/서킷브레이커),
``stream_worker`` (스트림 슬롯), ``config``/``constants``/``exceptions``/
``logging_config``/``pdf_utils``/``similarity``/``system_pressure`` (분리 유지).

구경로 shim(도메인 유틸 5종, 재시도, 서킷브레이커)은
순수 re-export 파일로 유지한다 (Phase 1 only, Task10에서 제거).
"""
