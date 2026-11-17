"""
IndusPredict AI - Security, Password Hashing & Role-Based Access Control (RBAC).
"""

from dataclasses import dataclass
from datetime import datetime
import secrets
from typing import Dict, List, Optional, Set
import bcrypt

from induspredict.core.utils import utc_now
from induspredict.core.config import (
    ROLE_ADMIN,
    ROLE_ENGINEER,
    ROLE_RESEARCHER,
    ROLE_VIEWER,
)

# Granular Permission Keys
PERM_USER_MANAGEMENT = "user:manage"
PERM_MACHINE_VIEW = "machine:view"
PERM_MACHINE_EDIT = "machine:edit"
PERM_MACHINE_DELETE = "machine:delete"
PERM_SENSOR_VIEW = "sensor:view"
PERM_SENSOR_EDIT = "sensor:edit"
PERM_PREDICTION_RUN = "prediction:run"
PERM_PREDICTION_VIEW = "prediction:view"
PERM_RUL_RUN = "rul:run"
PERM_RUL_VIEW = "rul:view"
PERM_SIMULATION_RUN = "simulation:run"
PERM_XAI_VIEW = "xai:view"
PERM_WORK_ORDER_MANAGE = "work_order:manage"
PERM_DRIFT_MANAGE = "drift:manage"
PERM_MODEL_RETRAIN = "model:retrain"
PERM_REPORT_EXPORT = "report:export"
PERM_AUDIT_VIEW = "audit:view"

# Role-to-Permissions Mapping
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    ROLE_ADMIN: {
        PERM_USER_MANAGEMENT,
        PERM_MACHINE_VIEW,
        PERM_MACHINE_EDIT,
        PERM_MACHINE_DELETE,
        PERM_SENSOR_VIEW,
        PERM_SENSOR_EDIT,
        PERM_PREDICTION_RUN,
        PERM_PREDICTION_VIEW,
        PERM_RUL_RUN,
        PERM_RUL_VIEW,
        PERM_SIMULATION_RUN,
        PERM_XAI_VIEW,
        PERM_WORK_ORDER_MANAGE,
        PERM_DRIFT_MANAGE,
        PERM_MODEL_RETRAIN,
        PERM_REPORT_EXPORT,
        PERM_AUDIT_VIEW,
    },
    ROLE_ENGINEER: {
        PERM_MACHINE_VIEW,
        PERM_MACHINE_EDIT,
        PERM_SENSOR_VIEW,
        PERM_SENSOR_EDIT,
        PERM_PREDICTION_RUN,
        PERM_PREDICTION_VIEW,
        PERM_RUL_RUN,
        PERM_RUL_VIEW,
        PERM_SIMULATION_RUN,
        PERM_XAI_VIEW,
        PERM_WORK_ORDER_MANAGE,
        PERM_REPORT_EXPORT,
    },
    ROLE_RESEARCHER: {
        PERM_MACHINE_VIEW,
        PERM_SENSOR_VIEW,
        PERM_PREDICTION_RUN,
        PERM_PREDICTION_VIEW,
        PERM_RUL_RUN,
        PERM_RUL_VIEW,
        PERM_SIMULATION_RUN,
        PERM_XAI_VIEW,
        PERM_DRIFT_MANAGE,
        PERM_MODEL_RETRAIN,
        PERM_REPORT_EXPORT,
    },
    ROLE_VIEWER: {
        PERM_MACHINE_VIEW,
        PERM_SENSOR_VIEW,
        PERM_PREDICTION_VIEW,
        PERM_RUL_VIEW,
        PERM_XAI_VIEW,
        PERM_REPORT_EXPORT,
    },
}


def hash_password(password: str) -> str:
    """Hash raw password using bcrypt with automatic salt."""
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify raw password against bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


@dataclass
class UserSession:
    """Represents an authenticated desktop user session."""
    user_id: int
    username: str
    full_name: str
    email: str
    role_name: str
    session_token: str
    login_time: datetime
    is_active: bool = True

    def has_permission(self, permission: str) -> bool:
        """Check if active user's role possesses the requested permission."""
        perms = ROLE_PERMISSIONS.get(self.role_name, set())
        return permission in perms


class SessionManager:
    """Thread-safe desktop session manager."""
    _current_session: Optional[UserSession] = None

    @classmethod
    def set_session(cls, user_id: int, username: str, full_name: str, email: str, role_name: str) -> UserSession:
        token = secrets.token_hex(24)
        session = UserSession(
            user_id=user_id,
            username=username,
            full_name=full_name,
            email=email,
            role_name=role_name,
            session_token=token,
            login_time=utc_now(),
            is_active=True,
        )
        cls._current_session = session
        return session

    @classmethod
    def get_current_session(cls) -> Optional[UserSession]:
        return cls._current_session

    @classmethod
    def clear_session(cls) -> None:
        cls._current_session = None

    @classmethod
    def is_authenticated(cls) -> bool:
        return cls._current_session is not None and cls._current_session.is_active

    @classmethod
    def require_permission(cls, permission: str) -> bool:
        session = cls.get_current_session()
        if not session or not session.is_active:
            return False
        return session.has_permission(permission)
