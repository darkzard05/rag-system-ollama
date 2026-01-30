"""
Task 22-1: RBAC (Role-Based Access Control) System
역할 기반 접근 제어 시스템
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class ActionType(Enum):
    """작업 타입"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"


class ResourceType(Enum):
    """리소스 타입"""

    DATA = "data"
    CONFIG = "config"
    USER = "user"
    LOG = "log"
    REPORT = "report"
    SYSTEM = "system"


@dataclass
class Permission:
    """권한"""

    permission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    action: ActionType = ActionType.READ
    resource: ResourceType = ResourceType.DATA
    resource_id: str | None = None  # 특정 리소스에 대한 권한
    created_at: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.permission_id)

    def __eq__(self, other):
        if isinstance(other, Permission):
            return self.permission_id == other.permission_id
        return False


@dataclass
class Role:
    """역할"""

    role_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    permissions: set[Permission] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    is_admin: bool = False

    def __hash__(self):
        return hash(self.role_id)

    def __eq__(self, other):
        if isinstance(other, Role):
            return self.role_id == other.role_id
        return False

    def add_permission(self, permission: Permission):
        """권한 추가"""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission):
        """권한 제거"""
        self.permissions.discard(permission)

    def has_permission(self, permission: Permission) -> bool:
        """권한 보유 확인"""
        if self.is_admin:
            return True
        return permission in self.permissions


@dataclass
class User:
    """사용자"""

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    roles: set[Role] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_login: float | None = None
    is_active: bool = True
    is_locked: bool = False
    failed_login_attempts: int = 0

    def __hash__(self):
        return hash(self.user_id)

    def __eq__(self, other):
        if isinstance(other, User):
            return self.user_id == other.user_id
        return False

    def add_role(self, role: Role):
        """역할 추가"""
        self.roles.add(role)

    def remove_role(self, role: Role):
        """역할 제거"""
        self.roles.discard(role)

    def has_role(self, role: Role) -> bool:
        """역할 보유 확인"""
        return role in self.roles


@dataclass
class AccessLog:
    """접근 로그"""

    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    action: ActionType = ActionType.READ
    resource: ResourceType = ResourceType.DATA
    resource_id: str | None = None
    allowed: bool = False
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class RBAC:
    """역할 기반 접근 제어"""

    def __init__(self):
        self._users: dict[str, User] = {}
        self._roles: dict[str, Role] = {}
        self._permissions: dict[str, Permission] = {}
        self._access_logs: list[AccessLog] = []
        self._max_logs = 100000
        self._lock = RLock()

        # 기본 역할 생성
        self._create_default_roles()

    def _create_default_roles(self):
        """기본 역할 생성"""
        # Admin 역할
        admin_role = Role(
            name="admin", description="Administrator with full access", is_admin=True
        )
        self._roles[admin_role.role_id] = admin_role

        # Editor 역할
        editor_role = Role(name="editor", description="Can read and write data")

        # Reader 역할
        reader_role = Role(name="reader", description="Read-only access")

        self._roles[editor_role.role_id] = editor_role
        self._roles[reader_role.role_id] = reader_role

    def create_permission(
        self,
        name: str,
        action: ActionType,
        resource: ResourceType,
        resource_id: str | None = None,
    ) -> str:
        """권한 생성"""
        with self._lock:
            permission = Permission(
                name=name, action=action, resource=resource, resource_id=resource_id
            )
            self._permissions[permission.permission_id] = permission
            return permission.permission_id

    def get_permission(self, permission_id: str) -> Permission | None:
        """권한 조회"""
        with self._lock:
            return self._permissions.get(permission_id)

    def create_role(
        self, name: str, description: str = "", is_admin: bool = False
    ) -> str:
        """역할 생성"""
        with self._lock:
            role = Role(name=name, description=description, is_admin=is_admin)
            self._roles[role.role_id] = role
            return role.role_id

    def get_role(self, role_id: str) -> Role | None:
        """역할 조회"""
        with self._lock:
            return self._roles.get(role_id)

    def get_role_by_name(self, name: str) -> Role | None:
        """이름으로 역할 조회"""
        with self._lock:
            for role in self._roles.values():
                if role.name == name:
                    return role
            return None

    def add_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """역할에 권한 추가"""
        with self._lock:
            role = self._roles.get(role_id)
            permission = self._permissions.get(permission_id)

            if not role or not permission:
                return False

            role.add_permission(permission)
            return True

    def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """역할에서 권한 제거"""
        with self._lock:
            role = self._roles.get(role_id)
            permission = self._permissions.get(permission_id)

            if not role or not permission:
                return False

            role.remove_permission(permission)
            return True

    def create_user(self, username: str, email: str) -> str:
        """사용자 생성"""
        with self._lock:
            user = User(username=username, email=email)
            self._users[user.user_id] = user
            return user.user_id

    def get_user(self, user_id: str) -> User | None:
        """사용자 조회"""
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """사용자명으로 사용자 조회"""
        with self._lock:
            for user in self._users.values():
                if user.username == username:
                    return user
            return None

    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """사용자에게 역할 할당"""
        with self._lock:
            user = self._users.get(user_id)
            role = self._roles.get(role_id)

            if not user or not role:
                return False

            user.add_role(role)
            return True

    def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """사용자에게서 역할 회수"""
        with self._lock:
            user = self._users.get(user_id)
            role = self._roles.get(role_id)

            if not user or not role:
                return False

            user.remove_role(role)
            return True

    def check_access(
        self,
        user_id: str,
        action: ActionType,
        resource: ResourceType,
        resource_id: str | None = None,
    ) -> bool:
        """접근 권한 확인"""
        with self._lock:
            user = self._users.get(user_id)

            if not user:
                self._log_access(
                    user_id,
                    action,
                    resource,
                    resource_id,
                    False,
                    {"reason": "user_not_found"},
                )
                return False

            if not user.is_active:
                self._log_access(
                    user_id,
                    action,
                    resource,
                    resource_id,
                    False,
                    {"reason": "user_inactive"},
                )
                return False

            if user.is_locked:
                self._log_access(
                    user_id,
                    action,
                    resource,
                    resource_id,
                    False,
                    {"reason": "user_locked"},
                )
                return False

            # 사용자의 모든 역할 확인
            for role in user.roles:
                if role.is_admin:
                    self._log_access(user_id, action, resource, resource_id, True)
                    return True

                for permission in role.permissions:
                    if (
                        permission.action == action
                        and permission.resource == resource
                        and (
                            permission.resource_id is None
                            or permission.resource_id == resource_id
                        )
                    ):
                        self._log_access(user_id, action, resource, resource_id, True)
                        return True

            self._log_access(
                user_id,
                action,
                resource,
                resource_id,
                False,
                {"reason": "no_permission"},
            )
            return False

    def _log_access(
        self,
        user_id: str,
        action: ActionType,
        resource: ResourceType,
        resource_id: str | None,
        allowed: bool,
        details: dict[str, Any] | None = None,
    ):
        """접근 로그 기록"""
        log = AccessLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            allowed=allowed,
            details=details or {},
        )

        self._access_logs.append(log)

        # 로그 크기 제한
        if len(self._access_logs) > self._max_logs:
            self._access_logs.pop(0)

    def get_access_logs(
        self,
        user_id: str | None = None,
        limit: int = 100,
        allowed_only: bool | None = None,
    ) -> list[AccessLog]:
        """접근 로그 조회"""
        with self._lock:
            logs = self._access_logs

            if user_id:
                logs = [log for log in logs if log.user_id == user_id]

            if allowed_only is not None:
                logs = [log for log in logs if log.allowed == allowed_only]

            # 최신순 정렬
            logs.sort(key=lambda x: x.timestamp, reverse=True)

            return logs[:limit]

    def lock_user(self, user_id: str) -> bool:
        """사용자 잠금"""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.is_locked = True
                return True
            return False

    def unlock_user(self, user_id: str) -> bool:
        """사용자 잠금 해제"""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.is_locked = False
                user.failed_login_attempts = 0
                return True
            return False

    def deactivate_user(self, user_id: str) -> bool:
        """사용자 비활성화"""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.is_active = False
                return True
            return False

    def activate_user(self, user_id: str) -> bool:
        """사용자 활성화"""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.is_active = True
                return True
            return False

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """사용자 권한 조회"""
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return set()

            permissions = set()
            for role in user.roles:
                if role.is_admin:
                    # Admin 역할이면 모든 권한
                    return set(self._permissions.values())
                permissions.update(role.permissions)

            return permissions

    def get_statistics(self) -> dict[str, Any]:
        """통계"""
        with self._lock:
            total_access_allowed = sum(1 for log in self._access_logs if log.allowed)
            total_access_denied = sum(1 for log in self._access_logs if not log.allowed)

            return {
                "total_users": len(self._users),
                "total_roles": len(self._roles),
                "total_permissions": len(self._permissions),
                "total_access_logs": len(self._access_logs),
                "access_allowed": total_access_allowed,
                "access_denied": total_access_denied,
                "denial_rate": (
                    total_access_denied / (total_access_allowed + total_access_denied)
                    if (total_access_allowed + total_access_denied) > 0
                    else 0
                ),
            }
