"""
A4: ClientPool 클라이언트 재사용 호스트 추적 검증.

수정 전에는 getattr(client, "base_url", "") != host 표현식이 항상 참으로 평가되어
(설치된 ollama.Client/AsyncClient에 base_url 속성이 없음) 매 호출마다 클라이언트가
재생성되었다. 이제 ClientPool이 생성 시점의 호스트를 _sync_host/_async_host로 직접
추적한다. 여기서는 호스트가 동일하면 동일 객체를 재사용하고, 호스트가 바뀌면
재생성되는지 검증한다.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.resource_manager import ClientPool


@pytest.fixture
def pool() -> ClientPool:
    return ClientPool()


def test_sync_client_reused_when_host_unchanged(pool: ClientPool) -> None:
    """동일 호스트로 두 번 호출하면 같은 객체를 반환하고 생성자는 한 번만 호출."""
    with patch("ollama.Client", side_effect=MagicMock) as client_cls:
        first = pool.get_sync_client("http://host-a:11434")
        second = pool.get_sync_client("http://host-a:11434")

        assert first is second
        client_cls.assert_called_once_with(host="http://host-a:11434")


def test_sync_client_recreated_when_host_changes(pool: ClientPool) -> None:
    """호스트 A → B 로 바뀌면 서로 다른 객체를 반환하고 생성자는 두 번 호출."""
    with patch("ollama.Client", side_effect=MagicMock) as client_cls:
        a = pool.get_sync_client("http://host-a:11434")
        b = pool.get_sync_client("http://host-b:11434")

        assert a is not b
        assert client_cls.call_count == 2


@pytest.mark.asyncio
async def test_async_client_reused_when_loop_and_host_unchanged(
    pool: ClientPool,
) -> None:
    """동일 루프·동일 호스트에서는 같은 AsyncClient 객체를 재사용."""
    with patch("ollama.AsyncClient", side_effect=MagicMock) as client_cls:
        first = await pool.get_async_client("http://host-a:11434")
        second = await pool.get_async_client("http://host-a:11434")

        assert first is second
        client_cls.assert_called_once_with(host="http://host-a:11434")
