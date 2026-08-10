"""
PDF 단어 좌표(word_coords)를 위한 사이드 캐시 매니저.
벡터 저장소의 메타데이터 비대화를 방지하기 위해 좌표 데이터를 SQLite에 별도로 저장하고 관리합니다.

이벤트 루프 독립성:
CoordCacheManager는 전용 owner 이벤트 루프 스레드를 보유합니다.
모든 async 진입점은 호출자 루프와 무관하게 owner 루프로 라우팅되므로,
AsyncWorker 루프(인덱싱)와 FastAPI 루프(쿼리) 등 서로 다른 루프에서
동시에 접근해도 "attached to a different loop" 오류가 발생하지 않습니다.
"""

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from collections.abc import Coroutine
from typing import Any

import aiosqlite
import orjson

from common.config import PROJECT_ROOT

MAX_CACHE_SIZE_MB = 500
CACHE_TTL_DAYS = 7
_EVICTION_INTERVAL_SECONDS = 1800  # 30 minutes
_EVICTION_BATCH_SIZE = 128  # [R1b-07] 크기 상한 초과 시 사이클당 삭제 행 수
_CONNECTION_MAX_AGE = 300  # 5분: 연결 최대 생존 시간 (초)
_CONNECTION_HEALTH_CHECK_INTERVAL = 30  # 30초: 헬스체크 간격
_WRITE_BEHIND_QUEUE_MAX = 5000  # 백그라운드 큐 최대 대기 항목 (메모리 바운드)

logger = logging.getLogger(__name__)

# 캐시 디렉토리 및 DB 설정
COORD_CACHE_DIR = PROJECT_ROOT / ".model_cache" / "coord_cache"
COORD_CACHE_DB = COORD_CACHE_DIR / "coords.db"


def to_coord5(word: Any) -> Any:
    """단어 좌표를 캐시 저장 규격 5-tuple ``(x0, y0, x1, y1, word)``로 정규화합니다.

    좌표 캐시에는 추출 경로별로 서로 다른 튜플 길이가 유입될 수 있어
    (pymupdf4llm ``extractWORDS()`` 8-tuple ``(x0, y0, x1, y1, word, block, line, word_no)``
    vs C-엔진/하이드레이터 5-tuple), 저장 전에 한 번에 통일합니다.
    dict 형식(테스트/기존 캐시 항목)은 그대로 통과시켜 하위 호환을 유지합니다.
    """
    if isinstance(word, (tuple, list)) and len(word) >= 5:
        return (word[0], word[1], word[2], word[3], word[4])
    return word


class CoordCacheManager:
    """단어 좌표 데이터를 SQLite에 캐싱하고 관리하는 클래스 (싱글톤)."""

    _instance: "CoordCacheManager | None" = None
    _owner_start_lock = threading.Lock()

    _conn: aiosqlite.Connection | None
    _conn_lock: asyncio.Lock
    _conn_created_at: float
    _last_health_check: float
    _conn_healthy: bool
    _queue: asyncio.Queue
    _worker_task: asyncio.Task | None
    _eviction_task: asyncio.Task | None
    _schema_ready: bool
    _eviction_started: bool
    _owner_loop: asyncio.AbstractEventLoop | None
    _owner_thread: threading.Thread | None

    def __new__(cls) -> "CoordCacheManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conn = None
            cls._instance._conn_lock = asyncio.Lock()
            cls._instance._conn_created_at = 0
            cls._instance._last_health_check = 0
            cls._instance._conn_healthy = False
            cls._instance._queue = asyncio.Queue(maxsize=_WRITE_BEHIND_QUEUE_MAX)
            cls._instance._worker_task = None
            cls._instance._eviction_task = None
            cls._instance._schema_ready = False
            cls._instance._eviction_started = False
            cls._instance._owner_loop = None
            cls._instance._owner_thread = None
        return cls._instance

    def _create_schema_sync(self) -> None:
        """DB 디렉토리 및 테이블을 동기로 생성합니다 (스레드 실행용)."""
        if not COORD_CACHE_DIR.exists():
            COORD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"좌표 캐시 디렉토리 생성: {COORD_CACHE_DIR}")

        with sqlite3.connect(COORD_CACHE_DB) as conn:
            _ = conn.execute("""
                CREATE TABLE IF NOT EXISTS coords (
                    file_hash TEXT,
                    page_num INTEGER,
                    coords BLOB,
                    created_at REAL,
                    PRIMARY KEY (file_hash, page_num)
                )
            """)
            _ = conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON coords(created_at)"
            )
            conn.commit()

    # -- owner 이벤트 루프 --------------------------------------------------

    def _ensure_owner_loop(self) -> None:
        """전용 owner 이벤트 루프 스레드를 1회 시작합니다."""
        if (
            self._owner_thread is not None
            and self._owner_thread.is_alive()
            and self._owner_loop is not None
            and self._owner_loop.is_running()
        ):
            return

        with self._owner_start_lock:
            if (
                self._owner_thread is not None
                and self._owner_thread.is_alive()
                and self._owner_loop is not None
                and self._owner_loop.is_running()
            ):
                return

            loop = asyncio.new_event_loop()
            self._owner_loop = loop
            ready = threading.Event()

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            self._owner_thread = threading.Thread(
                target=_run_loop, daemon=True, name="CoordCacheLoop"
            )
            self._owner_thread.start()
            if not ready.wait(timeout=5):
                # 스레드 시작 실패: 필드 리셋 후 명확한 예외 (비실행 루프에
                # run_coroutine_threadsafe를 던지는 사고 방지)
                self._owner_thread = None
                self._owner_loop = None
                raise RuntimeError("좌표 캐시 owner 이벤트 루프 시작 실패")
            logger.info("좌표 캐시 owner 이벤트 루프 시작됨")

    async def _submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """코루틴을 owner 루프에서 실행합니다. 호출자가 이미 owner 루프면 직접 실행."""
        self._ensure_owner_loop()
        assert self._owner_loop is not None

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._owner_loop:
            return await coro
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._owner_loop)
        )

    def _stop_owner_loop(self) -> None:
        """owner 이벤트 루프를 중지하고 스레드를 정리합니다.

        - owner 루프 스레드 자신이 호출하면 join을 생략합니다.
          (자기 스레드 join 시 RuntimeError 발생)
        - _owner_start_lock으로 시작(_ensure_owner_loop)과 직렬화하여
          close() 중 재시작 경합을 방지합니다.
        """
        with self._owner_start_lock:
            loop = self._owner_loop
            thread = self._owner_thread
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if thread is not None and threading.current_thread() is not thread:
                thread.join(timeout=5)
            self._owner_thread = None
            self._owner_loop = None

    # -- 내부 구현 (항상 owner 루프에서 실행) --------------------------------

    async def _ensure_schema(self) -> None:
        """첫 DB 접근 시 스키마를 1회 생성하고 eviction 루프를 시작합니다."""
        if self._schema_ready:
            return
        await asyncio.to_thread(self._create_schema_sync)
        self._schema_ready = True
        if not self._eviction_started:
            self._eviction_started = True
            self._eviction_task = asyncio.create_task(self.start_eviction_loop())
            logger.info("좌표 캐시 eviction 루프 시작됨")

    async def _ensure_worker_started(self) -> None:
        """백그라운드 워커가 시작되지 않았다면 시작합니다."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._write_behind_worker())
            logger.info("좌표 캐시 백그라운드 워커 시작됨")

    async def _write_behind_worker(self) -> None:
        """백그라운드에서 큐의 좌표 데이터를 SQLite에 저장합니다."""
        while True:
            try:
                file_hash, page_num, coords = await self._queue.get()
                try:
                    db = await self._get_connection()
                    await db.execute(
                        "INSERT OR REPLACE INTO coords (file_hash, page_num, coords, created_at) VALUES (?, ?, ?, ?)",
                        (file_hash, page_num, orjson.dumps(coords), time.time()),
                    )
                    await db.commit()
                except (aiosqlite.Error, OSError, RuntimeError) as e:
                    logger.error(
                        f"백그라운드 저장 실패 ({file_hash}, p{page_num}): {e}"
                    )
                    self._conn_healthy = False
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except (aiosqlite.Error, OSError) as e:
                logger.error(f"워커 루프 오류: {e}")

    async def _get_connection(self) -> aiosqlite.Connection:
        """재사용 가능한 단일 연결을 반환합니다. 지연 검증 + 상태 추적로 SELECT 1 핑을 제거합니다."""
        await self._ensure_schema()
        now = time.monotonic()
        async with self._conn_lock:
            # 1. 기존 연결이 있고, 생존 시간 내이며, 최근 헬스체크 통과 시 즉시 반환
            if (
                self._conn is not None
                and self._conn_healthy
                and (now - self._conn_created_at) < _CONNECTION_MAX_AGE
                and (now - self._last_health_check) < _CONNECTION_HEALTH_CHECK_INTERVAL
            ):
                return self._conn

            # 2. [R1b-09] 연결 수명(_CONNECTION_MAX_AGE) 초과 → 헬스체크 주기와
            #    무관하게 즉시 재연결 (DB 파일 외부 교체/삭제 감지 지연 제거)
            if (
                self._conn is not None
                and (now - self._conn_created_at) >= _CONNECTION_MAX_AGE
            ):
                logger.debug(
                    f"연결 수명 경과 ({(now - self._conn_created_at):.0f}s), 재연결"
                )
                with contextlib.suppress(Exception):
                    await self._conn.close()
                self._conn = None
                self._conn_healthy = False
                return await self._connect()

            # 3. 수명 내이지만 헬스체크 주기 도래 → 지연 검증 (SELECT 1)
            if self._conn is not None:
                if (now - self._last_health_check) >= _CONNECTION_HEALTH_CHECK_INTERVAL:
                    try:
                        await self._conn.execute("SELECT 1")
                        self._conn_healthy = True
                        self._last_health_check = now
                        return self._conn
                    except Exception:
                        logger.debug("연결 헬스체크 실패, 재연결 시도")
                        with contextlib.suppress(Exception):
                            await self._conn.close()
                        self._conn = None
                        self._conn_healthy = False
                        return await self._connect()
                # 4. 수명·헬스체크 모두 유효 → 이전 상태 신뢰하고 반환 (낙관적)
                return self._conn

            # 5. 연결이 없거나 재연결 후 → 신규 생성 + WAL 모드
            return await self._connect()

    async def _connect(self) -> aiosqlite.Connection:
        """새 SQLite 연결을 생성하고 WAL 모드로 초기화합니다. (owner 루프 전용)"""
        try:
            conn = await aiosqlite.connect(COORD_CACHE_DB)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-32768")  # 32MB 캐시
        except (aiosqlite.Error, OSError) as e:
            # [R1b-09] 재연결 실패를 명시적 예외로 표면화 (낙관적 반환 제거)
            raise RuntimeError(f"좌표 캐시 DB 연결 실패: {COORD_CACHE_DB} — {e}") from e
        self._conn = conn
        self._conn_created_at = time.monotonic()
        self._last_health_check = time.monotonic()
        self._conn_healthy = True
        return conn

    async def start_eviction_loop(self) -> None:
        """백그라운드에서 주기적으로 캐시를 정리합니다."""
        while True:
            await asyncio.sleep(_EVICTION_INTERVAL_SECONDS)
            try:
                await self._evict_old_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백그라운드 캐시 정리 중 오류: {e}")

    async def _evict_old_entries(self) -> None:
        """TTL 및 크기 제한에 따라 오래된 캐시 파일을 정리합니다."""
        db = await self._get_connection()
        # 1. TTL 만료 삭제
        ttl_seconds = CACHE_TTL_DAYS * 86400
        now = time.time()
        _ = await db.execute(
            "DELETE FROM coords WHERE ? - created_at > ?", (now, ttl_seconds)
        )

        # 2. 크기 제한 — [R1b-07] SUM(length(coords)) 기반 초과 감지 후 배치 삭제.
        #    상한을 초과하면 오래된 행부터 반복 삭제해 한 사이클에서 상한 이하로
        #    되돌립니다. (기존 LIMIT 10은 10행/30분 = 하루 480행에 불과해
        #    500MB 상한이 사실상 비작동이었습니다)
        #
        #    [오버 삭제 방지] 삭제 후보를 "created_at 오름차순 누적 바이트가
        #    현재 초과분 이하"인 행으로 한정해, 한 번의 배치가 보존해야 할
        #    최신 행까지 휩쓸지 않도록 합니다. 행 수는 _EVICTION_BATCH_SIZE로
        #    상한을 둡니다(사이클당 작업량 바운드).
        size_limit = MAX_CACHE_SIZE_MB * 1024 * 1024
        while True:
            async with db.execute(
                "SELECT COALESCE(SUM(length(coords)), 0) FROM coords"
            ) as cursor:
                row = await cursor.fetchone()
            total_size = row[0] if row else 0
            if total_size <= size_limit:
                break
            excess = total_size - size_limit
            cur = await db.execute(
                "DELETE FROM coords WHERE rowid IN ("
                "SELECT rowid FROM ("
                "  SELECT rowid,"
                "    SUM(length(coords)) OVER ("
                "      ORDER BY created_at ASC, rowid ASC"
                "    ) AS cum_size"
                "  FROM coords"
                ") WHERE cum_size <= ?"
                " ORDER BY rowid LIMIT ?)",
                (excess, _EVICTION_BATCH_SIZE),
            )
            if not cur.rowcount:
                # 누적 바이트가 초과분을 넘는 단일 행만 남은 경우(너무 큰 1행),
                # 가장 오래된 1행을 삭제해 진행을 보장합니다. 최신 행은 모든
                # 오래된 행이 사라지고도 상한을 넘을 때만 삭제 대상이 됩니다.
                cur = await db.execute(
                    "DELETE FROM coords WHERE rowid = ("
                    "SELECT rowid FROM coords"
                    " ORDER BY created_at ASC, rowid ASC LIMIT 1"
                    ")"
                )
                if not cur.rowcount:
                    # 안전망: 삭제할 행이 없는데도 상한 미달이 아닌 경우 루프 종료
                    break
        await db.commit()

    # -- 공개 API (루프 독립) ----------------------------------------------

    async def save_coords(
        self,
        file_hash: str,
        page_num: int,
        coords: list[Any],
    ) -> bool:
        """좌표 데이터를 캐시에 저장합니다 (Write-Behind)."""
        if not file_hash or not coords:
            return False

        # [R2-09] 저장 전 5-tuple 정규화 — pymupdf4llm 8-tuple과 C-엔진 5-tuple을 통일.
        # 하이드레이션 읽기 경로(utils.py: c[0..4] 소비)와 대칭을 유지합니다.
        normalized = [to_coord5(w) for w in coords]

        try:
            await self._submit(self._save_coords_impl(file_hash, page_num, normalized))
            return True
        except Exception as e:
            logger.error(f"좌표 캐시 큐 삽입 실패 ({file_hash}, p{page_num}): {e}")
            return False

    async def _save_coords_impl(
        self,
        file_hash: str,
        page_num: int,
        coords: list[dict[str, Any]],
    ) -> None:
        await self._ensure_worker_started()
        try:
            # 큐가 가득 차면 블로킹하지 않고 드롭 (좌표는 재계산 가능한 캐시)
            self._queue.put_nowait((file_hash, page_num, coords))
        except asyncio.QueueFull:
            logger.warning(
                f"좌표 캐시 큐 가득 참 — 좌표 저장 생략 ({file_hash}, p{page_num})"
            )

    async def get_coords_batch(
        self, file_hash: str, page_nums: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        """캐시에서 여러 페이지의 좌표 데이터를 한 번에 로드합니다."""
        if not page_nums:
            return {}

        try:
            return await self._submit(self._get_coords_batch_impl(file_hash, page_nums))
        except Exception as e:
            logger.error(f"좌표 캐시 배치 로드 실패 ({file_hash}): {e}")
            return {}

    async def _get_coords_batch_impl(
        self, file_hash: str, page_nums: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        try:
            # IN 절을 위한 쿼리 생성
            placeholders = ",".join(["?"] * len(page_nums))
            query = f"SELECT page_num, coords FROM coords WHERE file_hash = ? AND page_num IN ({placeholders})"

            results = {}
            db = await self._get_connection()
            async with db.execute(query, (file_hash, *page_nums)) as cursor:
                async for row in cursor:
                    results[row[0]] = orjson.loads(row[1])
            return results
        except Exception as e:
            logger.error(f"좌표 캐시 배치 로드 실패 ({file_hash}): {e}")
            self._conn_healthy = False
            return {}

    async def get_coords(
        self, file_hash: str, page_num: int
    ) -> list[dict[str, Any]] | None:
        """캐시에서 좌표 데이터를 로드합니다."""
        try:
            return await self._submit(self._get_coords_impl(file_hash, page_num))
        except Exception as e:
            logger.error(f"좌표 캐시 로드 실패 ({file_hash}, p{page_num}): {e}")
            return None

    async def _get_coords_impl(
        self, file_hash: str, page_num: int
    ) -> list[dict[str, Any]] | None:
        try:
            db = await self._get_connection()
            async with db.execute(
                "SELECT coords FROM coords WHERE file_hash = ? AND page_num = ?",
                (file_hash, page_num),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return orjson.loads(row[0])  # type: ignore
            return None
        except Exception as e:
            logger.error(f"좌표 캐시 로드 실패 ({file_hash}, p{page_num}): {e}")
            self._conn_healthy = False
            return None

    async def clear_cache(self, file_hash: str | None = None):
        """특정 파일 또는 전체 캐시를 삭제합니다 (비동기)."""
        await self._submit(self._clear_cache_impl(file_hash))

    async def _clear_cache_impl(self, file_hash: str | None = None):
        try:
            db = await self._get_connection()
            if file_hash:
                await db.execute("DELETE FROM coords WHERE file_hash = ?", (file_hash,))
            else:
                await db.execute("DELETE FROM coords")
            await db.commit()
        except aiosqlite.Error as e:
            logger.error(f"좌표 캐시 삭제 중 오류: {e}")

    async def close(self) -> None:
        """연결 및 백그라운드 태스크를 정리하고 owner 루프를 종료합니다.

        종료 후 재사용(테스트)을 위해 루프에 바인딩된 프리미티브도
        초기화합니다. (이전 루프에 묶인 _conn_lock/_queue는 다음 루프에서
        "different event loop" 오류를 유발합니다)
        """
        if self._owner_loop is None:
            return
        await self._submit(self._close_impl())
        self._stop_owner_loop()
        self._conn_lock = asyncio.Lock()
        self._queue = asyncio.Queue(maxsize=_WRITE_BEHIND_QUEUE_MAX)
        # [R1b-05] close→재사용 시 가드를 리셋하지 않으면 _ensure_schema가 조기
        # 반환해 eviction 루프가 영구 비활성됩니다 (worker는 _ensure_worker_started가
        # _worker_task is None으로 재시작하지만 eviction은 재시작되지 않았음).
        self._schema_ready = False
        self._eviction_started = False

    async def _close_impl(self) -> None:
        async with self._conn_lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    await self._conn.close()
                self._conn = None
                self._conn_healthy = False

        for task in (self._worker_task, self._eviction_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._worker_task = None
        self._eviction_task = None


# 싱글톤 인스턴스 노출
coord_cache = CoordCacheManager()
