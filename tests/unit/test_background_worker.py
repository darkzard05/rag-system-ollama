from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.common.utils import run_in_background_worker


@pytest.mark.asyncio
async def test_run_in_background_worker_success():
    """
    Verify that run_in_background_worker correctly calls request_rerun(None)
    after the background coroutine completes in the happy path.
    Uses the Streamlit 1.54.0 Runtime._session_mgr API path.
    """
    # Setup mocks
    mock_ctx = MagicMock()
    mock_ctx.session_id = "test_sid_abc"

    mock_runtime = MagicMock()
    mock_session_info = MagicMock()
    mock_session_info.session = MagicMock()
    mock_runtime._session_mgr.get_session_info.return_value = mock_session_info

    mock_coro = AsyncMock()
    session_id = "test_session_123"

    # Patch all dependencies
    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx") as mock_add_ctx, \
         patch("streamlit.runtime.get_instance", return_value=mock_runtime), \
         patch("core.session.SessionManager.set_session_id") as mock_set_sid, \
         patch("threading.Thread") as mock_thread_cls:

        run_in_background_worker(mock_coro, session_id)

        # Capture the target function passed to threading.Thread
        mock_thread_cls.assert_called_once()
        target = mock_thread_cls.call_args.kwargs['target']

        # Run the wrapper synchronously to verify its logic
        target()

        # Verifications
        mock_add_ctx.assert_called_once()
        mock_set_sid.assert_called_once_with(session_id)
        mock_runtime._session_mgr.get_session_info.assert_called_once_with(
            "test_sid_abc"
        )
        mock_session_info.session.request_rerun.assert_called_once_with(None)

@pytest.mark.asyncio
async def test_run_in_background_worker_no_ctx():
    """
    Verify that run_in_background_worker handles cases where get_script_run_ctx()
    returns None without raising an error.
    """
    mock_coro = AsyncMock()
    session_id = "test_session_456"

    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=None), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx"), \
         patch("streamlit.runtime.get_instance"), \
         patch("core.session.SessionManager.set_session_id"), \
         patch("threading.Thread") as mock_thread_cls:

        run_in_background_worker(mock_coro, session_id)
        target = mock_thread_cls.call_args.kwargs['target']

        # Should not raise when ctx is None — guard `if ctx and ctx.session_id`
        target()

@pytest.mark.asyncio
async def test_run_in_background_worker_no_session_id():
    """
    Verify that run_in_background_worker handles cases where ctx.session_id is
    None without raising an error.
    """
    mock_ctx = MagicMock()
    mock_ctx.session_id = None
    mock_coro = AsyncMock()
    session_id = "test_session_789"

    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx"), \
         patch("streamlit.runtime.get_instance"), \
         patch("core.session.SessionManager.set_session_id"), \
         patch("threading.Thread") as mock_thread_cls:

        run_in_background_worker(mock_coro, session_id)
        target = mock_thread_cls.call_args.kwargs['target']

        # Should not raise when ctx.session_id is None — guard `if ctx and ctx.session_id`
        target()
