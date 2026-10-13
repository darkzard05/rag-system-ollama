"""
중앙 집중식 로깅 설정 모듈.

전체 애플리케이션에서 일관된 로깅을 제공합니다:
- 콘솔 출력 (개발 중)
- 파일 저장 (프로덕션)
- 자동 로테이션 (용량 초과 시)
- 구조화된 형식 (분석 용이)
"""

import logging
import logging.handlers
import os
import shutil
import sys
import warnings
from pathlib import Path


class WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Windows 멀티프로세스 환경에서 안전한 로그 롤오버 핸들러.

    기본 ``RotatingFileHandler.doRollover`` 는 ``os.rename(app.log, app.log.1)`` 을
    사용하는데, Windows 에서는 같은 파일을 다른 프로세스(예: 2번째 Streamlit
    인스턴스)가 열고 있으면 ``WinError 32 (ERROR_SHARING_VIOLATION)`` 로 실패합니다.

    이를 해결하기 위해 rename 대신 ``copy + truncate`` 전략을 사용합니다.
    원본 파일을 다른 프로세스가 열고 있어도 내용을 복사한 뒤 원본을 0바이트로
    자르므로 모든 writer 의 핸들이 유효하게 유지됩니다.
    """

    def rotate(self, source: str, dest: str) -> None:
        """원본을 대상으로 복사한 뒤 원본을 비웁니다 (Windows 공유 위반 회피)."""
        if not os.path.exists(source):
            return
        try:
            shutil.copyfile(source, dest)
            with open(source, "r+b") as fh:
                fh.truncate(0)
        except OSError as exc:
            # WinError 32 = 파일을 다른 프로세스가 사용 중. 롤오버를 보류하고
            # 다음 emit 에서 재시도하도록 경고만 남깁니다 (무음 삭제 금지).
            if getattr(exc, "winerror", None) == 32 or exc.errno in (13,):
                logging.getLogger(__name__).warning(
                    "로그 롤오버 보류: %s 를 다른 프로세스가 사용 중 (다음 회차 재시도)",
                    source,
                )
                return
            raise


class ContextFilter(logging.Filter):
    """추가 컨텍스트 정보를 로그에 포함하는 필터"""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        로그 레코드에 컨텍스트 정보 추가.

        Args:
            record: 로그 레코드

        Returns:
            True (항상 필터를 통과)
        """
        # 함수명과 라인 번호는 이미 포함됨
        return True


def _preimport_noisy_transformers_processors() -> None:
    """auto_docstring print 스팸을 유발하는 리프 모듈을 선행 import한다.

    transformers 리프 이미지 프로세서(deepseek_vl_hybrid / kimi_k25 /
    paddleocr_vl)의 ``@auto_docstring`` 데코레이터는 클래스 정의(import) 시점에
    미문서 kwargs를 ``print`` 로 stdout에 뿌린다 (logging/warnings 아님).
    파이프라인 빌드 중 lazy import로 실행되면 터미널 로그를 오염시키므로,
    로깅 설정 시점(앱 시작, 단일 스레드)에 stdout 억제 하에 미리 import해
    데코레이터를 소진시킨다. 이후 import는 캐시 적중이라 print가 재발하지
    않는다. 프로젝트 코드는 stdout print를 사용하지 않으므로(전부 stderr
    로깅) 억제 윈도우가 정상 출력을 삼킬 우려가 없다.

    - pytest 하에서는 스킵한다 (단위 테스트는 해당 경로를 타지 않으며,
      매 테스트 프로세스마다 수 초의 import 비용을 치를 이유가 없다).
    - 모듈명 변경 등 실패는 전부 흡수한다 (fail-open: 억제 실패 시 스팸이
      그대로 출력될 뿐 기능에 영향 없음).
    """
    if "pytest" in sys.modules:
        return
    try:
        import contextlib
        import importlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            for _mod in (
                "transformers.models.deepseek_vl_hybrid.image_processing_deepseek_vl_hybrid",
                "transformers.models.deepseek_vl_hybrid.image_processing_pil_deepseek_vl_hybrid",
                "transformers.models.kimi_k25.image_processing_kimi_k25",
                "transformers.models.paddleocr_vl.image_processing_paddleocr_vl",
                "transformers.models.paddleocr_vl.image_processing_pil_paddleocr_vl",
            ):
                with contextlib.suppress(Exception):
                    importlib.import_module(_mod)
    except Exception:
        pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    애플리케이션 로깅을 설정합니다.

    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 파일로 저장 안함)
        console_output: 콘솔 출력 여부

    Returns:
        설정된 루트 로거

    Examples:
        >>> logger = setup_logging(log_level="INFO", log_file=Path("logs/app.log"))
        >>> logger.info("애플리케이션 시작")
    """

    # 루트 로거 가져오기
    root_logger = logging.getLogger()

    # ---- Warnings filtering (reduce noise in console/logs) ----
    # [최적화] Streamlit bare mode 실행 시 발생하는 무의미한 경고 전역 차단
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    logging.getLogger(
        "streamlit.runtime.scriptrunner_utils.script_run_context"
    ).setLevel(logging.ERROR)

    # NOTE: Some LangChain/LangGraph APIs emit beta warnings on each run.
    # We filter by message to avoid importing optional warning classes.
    warnings.filterwarnings(
        "ignore",
        message=r".*This API is in beta and may change in the future\..*",
    )

    # ---- External Libraries Noise Control ----
    # HTTP 요청 관련 로그가 너무 많으므로 WARNING 이상만 출력하도록 제어
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    # LangChain 관련 장황한 로그 제어
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # [추가] 캐시 보안 로그 폭사 차단 (가장 심각한 병목 해결)
    logging.getLogger("security").setLevel(logging.WARNING)
    logging.getLogger("security.cache_security").setLevel(logging.WARNING)

    # [추가] 최적화 및 모니터링 관련 반복 로그 차단 (인덱싱 성능 향상)
    logging.getLogger("services.optimization").setLevel(logging.WARNING)
    logging.getLogger("services.monitoring").setLevel(logging.WARNING)
    logging.getLogger("core.resource_pool").setLevel(logging.WARNING)

    # [정정] transformers 이미지 프로세서 미문서 키 출력(high_res_size /
    # merge_size / min_pixels / max_pixels 12연발)은 logging ERROR도 warnings도
    # 아닌 auto_docstring 데코레이터의 import 시점 stdout print다
    # (transformers/utils/auto_docstring.py — 공식 문서는 기여자용 CI 진단으로만
    # 의도, 업스트림 main에도 잔존). 따라서 로거 레벨/warnings 필터는 닿지
    # 않는다. 실제 차단은 아래 _preimport_noisy_transformers_processors()의
    # 선행 import + stdout 억제로 수행한다. 여기서는 통상 수준만 유지한다.
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    _preimport_noisy_transformers_processors()

    # [Phase A1] pymupdf4llm / RapidOCR 플러그인 print성 노이즈의 로거 경로 차단.
    # (to_markdown() 기본 자동 OCR 경로가 범위 병렬 수만큼 증폭됨.
    # stdout print 자체는 document_processor 래퍼에서 별도 처리.)
    logging.getLogger("pymupdf").setLevel(logging.WARNING)
    logging.getLogger("pymupdf4llm").setLevel(logging.WARNING)
    logging.getLogger("rapidocr_onnxruntime").setLevel(logging.WARNING)

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 로그 레벨 설정
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level_int)

    # 포맷터 정의 (표준화된 구분자 사용)
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - [%(name)s] [%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(message)s", datefmt="%H:%M:%S"
    )

    # 1. 콘솔 핸들러 (개발용 - 간단한 형식)
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(simple_formatter)
        console_handler.addFilter(ContextFilter())
        root_logger.addHandler(console_handler)

    # 2. 파일 핸들러 (프로덕션용 - 상세 형식)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = WindowsSafeRotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # 최대 5개 백업
            encoding="utf-8",
        )
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(ContextFilter())
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거를 가져옵니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        해당 모듈의 로거

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("모듈 시작")
    """
    return logging.getLogger(name)


# 기본 설정 (모듈 import 시 자동 실행)
if not logging.getLogger().handlers:
    setup_logging(log_level="INFO")
