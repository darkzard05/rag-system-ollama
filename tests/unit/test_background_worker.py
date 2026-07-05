from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.common.utils import run_in_background_worker


@pytest.mark.asyncio
async def test_run_in_background_worker_success():
    """
    Verify that run_in_background_worker correctly calls request_rerun(None)
    after the background coroutine completes in the happy path.
    """
    # Setup mocks
    mock_ctx = MagicMock()
    mock_ctx.session = MagicMock()
    mock_coro = AsyncMock()
    session_id = "test_session_123"

    # Patch modules imported inside the function
    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx") as mock_add_ctx, \
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
        mock_ctx.session.request_rerun.assert_called_once_with(None)

@pytest.mark.asyncio
async def test_run_in_background_worker_no_ctx():
    """
    Verify that run_in_background_worker handles cases where get_script_run_ctx()
    returns None without raising an AttributeError.
    """
    # Setup mocks
    mock_coro = AsyncMock()
    session_id = "test_session_456"

    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=None), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx"), \
         patch("core.session.SessionManager.set_session_id"), \
         patch("threading.Thread") as mock_thread_cls:

        run_in_background_worker(mock_coro, session_id)
        target = mock_thread_cls.call_args.kwargs['target']

        # Should not raise AttributeError when ctx is None
        target()

@pytest.mark.asyncio
async def test_run_in_background_worker_no_session():
    """
    Verify that run_in_background_worker handles cases where ctx.session is None
    without raising an AttributeError.
    """
    # Setup mocks
    mock_ctx = MagicMock()
    mock_ctx.session = None
    mock_coro = AsyncMock()
    session_id = "test_session_789"

    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx), \
         patch("streamlit.runtime.scriptrunner.add_script_run_ctx"), \
         patch("core.session.SessionManager.set_session_id"), \
         patch("threading.Thread") as mock_thread_cls:

        run_in_background_worker(mock_coro, session_id)
        target = mock_thread_cls.call_args.kwargs['target']

        # Should not raise AttributeError when ctx.session is None
        target()
