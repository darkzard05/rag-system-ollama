import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_directory(dir_path: Path, days: int, dry_run: bool = False, extensions: list = None):
    """지정된 디렉토리에서 일정 기간이 지난 파일을 삭제합니다."""
    if not dir_path.exists():
        logger.warning(f"디렉토리가 존재하지 않습니다: {dir_path}")
        return

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    count = 0
    total_size = 0

    logger.info(f"[{dir_path.name}] 청소 시작 (기준: {days}일 이전 파일)")

    # 디렉토리 내 모든 파일 순회 (재귀적)
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            # 확장자 필터링
            if extensions and file_path.suffix.lower() not in extensions:
                continue

            # 수정 시간 확인
            file_mtime = file_path.stat().st_mtime
            if file_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                last_mod = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                if dry_run:
                    logger.info(f"[DRY-RUN] 삭제 대상: {file_path.relative_to(ROOT_DIR)} (마지막 수정: {last_mod})")
                else:
                    try:
                        file_path.unlink()
                        logger.info(f"삭제됨: {file_path.relative_to(ROOT_DIR)}")
                    except Exception as e:
                        logger.error(f"삭제 실패: {file_path} - {e}")
                        continue
                
                count += 1
                total_size += file_size

    status = "발견됨" if dry_run else "삭제됨"
    logger.info(f"[{dir_path.name}] 완료: {count}개 파일 {status} (총 용량: {total_size / 1024 / 1024:.2f} MB)")
    return count

def main():
    parser = argparse.ArgumentParser(description="오래된 로그 및 리포트 파일을 정리합니다.")
    parser.add_argument("--days", type=int, default=7, help="삭제 기준 일수 (기본값: 7일)")
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제를 수행하지 않고 대상만 출력합니다.")
    parser.add_argument("--all", action="store_true", help="모든 아티팩트(로그, 리포트)를 정리합니다.")
    
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = [
            (ROOT_DIR / "logs", 7, None),
            (ROOT_DIR / "reports", 14, [".md", ".csv", ".json", ".png"]),
            (ROOT_DIR / "data" / "temp", 1, None)
        ]
    else:
        # 기본 타겟 설정
        targets = [
            (ROOT_DIR / "logs", args.days, None),
            (ROOT_DIR / "reports", args.days * 2, None) # 리포트는 조금 더 오래 보관
        ]

    total_deleted = 0
    for dir_path, days, ext in targets:
        total_deleted += clean_directory(dir_path, days, args.dry_run, ext)

    if not args.dry_run:
        logger.info(f"총 {total_deleted}개의 파일이 정리되었습니다.")
    else:
        logger.info(f"총 {total_deleted}개의 삭제 대상 파일이 확인되었습니다. (DRY-RUN 모드)")

if __name__ == "__main__":
    main()
