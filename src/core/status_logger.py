"""
시스템 작업 상태 로그를 관리하는 모듈.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StatusLogger:
    """
    세션별 작업 상태 로그를 관리합니다.
    """

    @staticmethod
    def add_status_log(session_manager: Any, msg: str, session_id: str | None = None):
        """작업 로그를 추가하고 채팅 이력에도 누적합니다. (최신 30개 보관)"""
        state = session_manager._get_state(session_id=session_id)
        if "status_logs" not in state:
            state["status_logs"] = []

        if state["status_logs"] and state["status_logs"][-1] == msg:
            return

        state["status_logs"].append(msg)
        session_manager.add_message(
            role="system", content=msg, msg_type="log", session_id=session_id
        )

        if len(state["status_logs"]) > 30:
            state["status_logs"] = state["status_logs"][-30:]

    @staticmethod
    def replace_last_status_log(
        session_manager: Any, msg: str, session_id: str | None = None
    ):
        """가장 최근의 로그를 찾아 결과로 교체합니다."""
        state = session_manager._get_state(session_id=session_id)
        if "status_logs" not in state or not state["status_logs"]:
            state["status_logs"] = [msg]
        else:
            state["status_logs"][-1] = msg

        messages = state.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("msg_type") == "log":
                old_content = messages[i].get("content", "")
                if (
                    "중..." in old_content
                    or "탐색" in old_content
                    or "분석" in old_content
                ):
                    messages[i]["content"] = msg
                    messages[i]["timestamp"] = time.time()
                    return
                break

        session_manager.add_message(
            role="system", content=msg, msg_type="log", session_id=session_id
        )
