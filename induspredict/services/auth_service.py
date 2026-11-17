"""
IndusPredict AI - Authentication & User Management Service.
Handles credential verification, secure session tokens, role checks, and audit logging.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import joinedload

from induspredict.core.utils import utc_now
from induspredict.core.logger import get_logger
from induspredict.core.security import (
    PERM_USER_MANAGEMENT,
    SessionManager,
    UserSession,
    hash_password,
    verify_password,
)
from induspredict.db.models import AuditLog, Role, User
from induspredict.db.session import get_db_session

logger = get_logger("services.auth")


class AuthService:
    """Service providing user authentication, registration, and RBAC enforcement."""

    @staticmethod
    def authenticate(username: str, password: str) -> Optional[UserSession]:
        """Verify credentials and establish user session upon successful match."""
        if not username or not password:
            return None

        with get_db_session() as session:
            user = (
                session.query(User)
                .options(joinedload(User.role))
                .filter(User.username == username.strip(), User.is_active.is_(True))
                .first()
            )

            if not user or not verify_password(password, user.password_hash):
                logger.warning(f"Failed login attempt for username: {username}")
                # Audit log for failed attempt
                audit = AuditLog(
                    username=username,
                    action="LOGIN_FAILED",
                    entity_type="User",
                    details="Invalid username or password supplied.",
                    timestamp=utc_now(),
                )
                session.add(audit)
                return None

            # Update last login timestamp
            user.last_login = utc_now()

            # Record successful login audit
            audit = AuditLog(
                user_id=user.id,
                username=user.username,
                action="LOGIN_SUCCESS",
                entity_type="User",
                entity_id=str(user.id),
                details=f"User successfully logged in with role '{user.role.name}'.",
                timestamp=utc_now(),
            )
            session.add(audit)

            # Create desktop session
            user_session = SessionManager.set_session(
                user_id=user.id,
                username=user.username,
                full_name=user.full_name,
                email=user.email,
                role_name=user.role.name,
            )
            logger.info(f"User '{username}' logged in successfully as '{user.role.name}'.")
            return user_session

    @staticmethod
    def register_user(
        username: str,
        email: str,
        password: str,
        full_name: str,
        role_name: str,
        admin_session: Optional[UserSession] = None,
    ) -> Dict[str, Any]:
        """Register a new user. Requires Administrator session when users already exist."""
        username = username.strip()
        email = email.strip().lower()
        full_name = full_name.strip()

        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters long."}
        if "@" not in email or "." not in email:
            return {"success": False, "error": "Invalid email address format."}
        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters long."}

        with get_db_session() as session:
            # Check RBAC if admin session is required
            user_count = session.query(User).count()
            if user_count > 0:
                if not admin_session or not admin_session.has_permission(PERM_USER_MANAGEMENT):
                    return {"success": False, "error": "Permission denied: Administrator privileges required."}

            # Check duplicates
            existing_user = session.query(User).filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                return {"success": False, "error": "Username or email is already registered."}

            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                return {"success": False, "error": f"Role '{role_name}' does not exist."}

            new_user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                full_name=full_name,
                role_id=role.id,
                is_active=True,
                created_at=utc_now(),
            )
            session.add(new_user)
            session.flush()

            # Audit log
            audit = AuditLog(
                user_id=admin_session.user_id if admin_session else new_user.id,
                username=admin_session.username if admin_session else username,
                action="USER_REGISTERED",
                entity_type="User",
                entity_id=str(new_user.id),
                details=f"Registered new user '{username}' with role '{role_name}'.",
                timestamp=utc_now(),
            )
            session.add(audit)

            logger.info(f"User '{username}' registered with role '{role_name}'.")
            return {
                "success": True,
                "user_id": new_user.id,
                "username": new_user.username,
                "role": role.name,
            }

    @staticmethod
    def logout() -> None:
        """Clear active user session and record audit log."""
        session = SessionManager.get_current_session()
        if session:
            try:
                with get_db_session() as db_session:
                    audit = AuditLog(
                        user_id=session.user_id,
                        username=session.username,
                        action="LOGOUT",
                        entity_type="User",
                        entity_id=str(session.user_id),
                        details="User logged out of desktop application.",
                        timestamp=utc_now(),
                    )
                    db_session.add(audit)
            except Exception as e:
                logger.error(f"Error logging logout audit: {e}")

        SessionManager.clear_session()
        logger.info("User session terminated.")

    @staticmethod
    def list_users() -> List[Dict[str, Any]]:
        """List registered users with roles."""
        with get_db_session() as session:
            users = session.query(User).options(joinedload(User.role)).all()
            return [
                {
                    "id": u.id,
                    "username": u.username,
                    "full_name": u.full_name,
                    "email": u.email,
                    "role": u.role.name if u.role else "Unknown",
                    "is_active": u.is_active,
                    "last_login": u.last_login.strftime("%Y-%m-%d %H:%M:%S") if u.last_login else "Never",
                    "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else "",
                }
                for u in users
            ]
