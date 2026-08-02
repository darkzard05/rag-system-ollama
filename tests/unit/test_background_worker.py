from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.utils import run_in_background_worker


class TestRunInBackgroundWorker:
    def test_submits_coroutine_to_async_worker(self):
        mock_ctx = MagicMock()
        mock_ctx.session_id = "test_sid"
        mock_coro = AsyncMock()
        mock_future = MagicMock()

        with (
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=mock_ctx,
            ),
            patch("common.async_worker.AsyncWorker") as mock_worker_cls,
        ):
            mock_worker = MagicMock()
            mock_worker.submit.return_value = mock_future
            mock_worker_cls.return_value = mock_worker

            run_in_background_worker(mock_coro, "test_session")

            mock_worker.submit.assert_called_once_with(mock_coro)
            mock_future.add_done_callback.assert_called_once()

    def test_callback_triggers_rerun_on_success(self):
        mock_ctx = MagicMock()
        mock_ctx.session_id = "test_sid"
        mock_runtime = MagicMock()
        mock_session_info = MagicMock()
        mock_runtime._session_mgr.get_session_info.return_value = mock_session_info
        mock_future = MagicMock()

        with (
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=mock_ctx,
            ),
            patch("common.async_worker.AsyncWorker") as mock_worker_cls,
        ):
            mock_worker = MagicMock()
            mock_worker.submit.return_value = mock_future
            mock_worker_cls.return_value = mock_worker

            run_in_background_worker(AsyncMock(), "test_sid")

            callback = mock_future.add_done_callback.call_args[0][0]

            with patch("streamlit.runtime.get_instance", return_value=mock_runtime):
                callback(mock_future)

            mock_session_info.session.request_rerun.assert_called_once_with(None)

    def test_callback_handles_no_ctx(self):
        mock_future = MagicMock()

        with (
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=None,
            ),
            patch("common.async_worker.AsyncWorker") as mock_worker_cls,
        ):
            mock_worker = MagicMock()
            mock_worker.submit.return_value = mock_future
            mock_worker_cls.return_value = mock_worker

            run_in_background_worker(AsyncMock(), "test_sid")

            callback = mock_future.add_done_callback.call_args[0][0]
            callback(mock_future)

    def test_callback_handles_none_session_id(self):
        mock_ctx = MagicMock()
        mock_ctx.session_id = None
        mock_future = MagicMock()

        with (
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=mock_ctx,
            ),
            patch("common.async_worker.AsyncWorker") as mock_worker_cls,
        ):
            mock_worker = MagicMock()
            mock_worker.submit.return_value = mock_future
            mock_worker_cls.return_value = mock_worker

            run_in_background_worker(AsyncMock(), "test_sid")

            callback = mock_future.add_done_callback.call_args[0][0]
            callback(mock_future)

    def test_callback_handles_error_without_raising(self):
        mock_ctx = MagicMock()
        mock_ctx.session_id = "test_sid"
        mock_future = MagicMock()
        error = RuntimeError("async task failed")
        mock_future.result.side_effect = error

        with (
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=mock_ctx,
            ),
            patch("common.async_worker.AsyncWorker") as mock_worker_cls,
            patch("streamlit.runtime.get_instance", return_value=None),
        ):
            mock_worker = MagicMock()
            mock_worker.submit.return_value = mock_future
            mock_worker_cls.return_value = mock_worker

            run_in_background_worker(AsyncMock(), "test_sid")

            callback = mock_future.add_done_callback.call_args[0][0]

            callback(mock_future)
