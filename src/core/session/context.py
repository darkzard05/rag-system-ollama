# 세션 식별자를 ContextVar를 통해 관리하는 컨텍스트 매니저
from contextvars import ContextVar

# [핵심] 전역 컨텍스트 변수 정의
_current_session_id: ContextVar[str | None] = ContextVar(
    "current_session_id", default=None
)


class ContextManager:
    @staticmethod
    def get_current_session_id() -> str | None:
        return _current_session_id.get()

    @staticmethod
    def set_current_session_id(session_id: str | None):
        _current_session_id.set(session_id)
