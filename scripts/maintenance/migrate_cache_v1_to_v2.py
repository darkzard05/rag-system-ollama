#!/usr/bin/env python
"""
캐시 마이그레이션 스크립트: v1 (레거시 pickle) → v2 (메타데이터 포함)

사용 방법:
    python scripts/migrate_cache_v1_to_v2.py --cache-dir .model_cache
    python scripts/migrate_cache_v1_to_v2.py --cache-dir .model_cache --dry-run
    python scripts/migrate_cache_v1_to_v2.py --cache-dir .model_cache --backup
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 상위 디렉터리를 path에 추가하여 src 모듈 임포트 가능하게
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.cache_security import CacheSecurityManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CacheMigrator:
    """캐시 v1 → v2 마이그레이션 관리자"""

    def __init__(
        self,
        cache_dir: str,
        backup: bool = True,
        dry_run: bool = False,
    ):
        """
        마이그레이터 초기화.

        Args:
            cache_dir: 캐시 디렉터리 경로
            backup: 마이그레이션 전 백업 여부
            dry_run: 실제 작업 없이 시뮬레이션만 수행
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.backup = backup
        self.dry_run = dry_run
        self.security_manager = CacheSecurityManager(security_level="medium")

        if not self.cache_dir.exists():
            raise ValueError(f"캐시 디렉터리가 없습니다: {self.cache_dir}")

        logger.info(f"마이그레이션 대상 디렉터리: {self.cache_dir}")
        logger.info(f"드라이런 모드: {dry_run}")
        logger.info(f"백업 생성: {backup}")

    def find_bm25_caches(self) -> list[Path]:
        """
        BM25 캐시 파일 찾기.

        Returns:
            BM25 pickle 파일 경로 리스트
        """
        bm25_files = list(self.cache_dir.rglob("bm25_retriever.pkl"))
        logger.info(f"찾은 BM25 캐시: {len(bm25_files)}개")
        return bm25_files

    def has_metadata(self, bm25_file: Path) -> bool:
        """
        메타데이터 파일 존재 여부 확인.

        Args:
            bm25_file: BM25 pickle 파일 경로

        Returns:
            메타데이터 파일 존재 여부
        """
        metadata_file = Path(str(bm25_file) + ".meta")
        return metadata_file.exists()

    def migrate_single_cache(self, bm25_file: Path) -> tuple[bool, str]:
        """
        단일 캐시 마이그레이션 수행.

        Args:
            bm25_file: BM25 pickle 파일 경로

        Returns:
            (성공 여부, 메시지)
        """
        try:
            # 이미 v2라면 스킵
            if self.has_metadata(bm25_file):
                return True, "이미 v2 캐시입니다 (스킵)"

            # 메타데이터 생성
            logger.debug(f"메타데이터 생성 중: {bm25_file}")
            metadata = self.security_manager.create_metadata_for_file(
                str(bm25_file),
                description=f"Migrated from v1 on {datetime.now().isoformat()}",
            )

            metadata_path = str(bm25_file) + ".meta"

            if not self.dry_run:
                self.security_manager.save_cache_metadata(metadata_path, metadata)
                logger.info(f"✓ 메타데이터 생성: {metadata_path}")
            else:
                logger.info(f"[DRY-RUN] 메타데이터 생성 예정: {metadata_path}")

            return True, "성공적으로 마이그레이션됨"

        except Exception as e:
            error_msg = f"마이그레이션 실패: {e}"
            logger.error(error_msg)
            return False, error_msg

    def create_backup(self) -> Path:
        """
        캐시 디렉터리 백업 생성.

        Returns:
            백업 디렉터리 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.cache_dir.parent / f"{self.cache_dir.name}_backup_{timestamp}"

        logger.info(f"백업 생성 중: {backup_dir}")
        shutil.copytree(self.cache_dir, backup_dir)
        logger.info(f"✓ 백업 완료: {backup_dir}")

        return backup_dir

    def migrate(self) -> tuple[int, int, list[tuple[Path, str]]]:
        """
        전체 캐시 마이그레이션 수행.

        Returns:
            (성공 개수, 실패 개수, 오류 리스트)
        """
        logger.info("=" * 60)
        logger.info("캐시 마이그레이션 시작 (v1 → v2)")
        logger.info("=" * 60)

        # 백업 생성
        if self.backup and not self.dry_run:
            self.create_backup()

        # BM25 캐시 찾기
        bm25_files = self.find_bm25_caches()
        if not bm25_files:
            logger.warning("마이그레이션할 BM25 캐시를 찾을 수 없습니다")
            return 0, 0, []

        # 마이그레이션
        success_count = 0
        failure_count = 0
        errors = []

        for i, bm25_file in enumerate(bm25_files, 1):
            logger.info(f"\n[{i}/{len(bm25_files)}] 처리 중: {bm25_file}")

            success, message = self.migrate_single_cache(bm25_file)

            if success:
                success_count += 1
                logger.info(f"  → {message}")
            else:
                failure_count += 1
                logger.error(f"  → {message}")
                errors.append((bm25_file, message))

        # 결과 출력
        logger.info("\n" + "=" * 60)
        logger.info("마이그레이션 완료")
        logger.info("=" * 60)
        logger.info(f"성공: {success_count}/{len(bm25_files)}")
        logger.info(f"실패: {failure_count}/{len(bm25_files)}")

        if errors:
            logger.warning("\n오류 목록:")
            for file_path, error_msg in errors:
                logger.warning(f"  - {file_path}")
                logger.warning(f"    {error_msg}")

        return success_count, failure_count, errors

    @staticmethod
    def verify_migration(cache_dir: Path) -> bool:
        """
        마이그레이션 결과 검증.

        Args:
            cache_dir: 캐시 디렉터리 경로

        Returns:
            검증 성공 여부
        """
        logger.info("\n검증 시작...")

        bm25_files = list(cache_dir.rglob("bm25_retriever.pkl"))
        missing_metadata = []

        for bm25_file in bm25_files:
            metadata_file = Path(str(bm25_file) + ".meta")
            if not metadata_file.exists():
                missing_metadata.append(bm25_file)

        if missing_metadata:
            logger.error(f"메타데이터가 없는 파일 {len(missing_metadata)}개:")
            for f in missing_metadata:
                logger.error(f"  - {f}")
            return False

        logger.info(f"✓ 모든 BM25 캐시의 메타데이터 확인됨 ({len(bm25_files)}개)")
        return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="캐시 마이그레이션: v1 (pickle) → v2 (메타데이터 포함)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 일반 마이그레이션
  python migrate_cache_v1_to_v2.py --cache-dir .model_cache

  # 드라이런 (실제 작업 없음)
  python migrate_cache_v1_to_v2.py --cache-dir .model_cache --dry-run

  # 백업 없이 마이그레이션
  python migrate_cache_v1_to_v2.py --cache-dir .model_cache --no-backup
        """,
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="캐시 디렉터리 경로 (기본: .model_cache)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="실제 작업 없이 시뮬레이션만 수행"
    )

    parser.add_argument(
        "--no-backup", action="store_true", help="마이그레이션 전 백업 생성 안함"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="마이그레이션 검증만 수행 (작업 없음)",
    )

    args = parser.parse_args()

    try:
        migrator = CacheMigrator(
            cache_dir=args.cache_dir,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )

        if args.verify_only:
            # 검증만 수행
            success = CacheMigrator.verify_migration(migrator.cache_dir)
            return 0 if success else 1
        else:
            # 마이그레이션 수행
            success_count, failure_count, errors = migrator.migrate()

            # 검증
            if success_count > 0:
                CacheMigrator.verify_migration(migrator.cache_dir)

            # 종료 코드
            return 0 if failure_count == 0 else 1

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
