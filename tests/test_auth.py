"""
Tests for IndusPredict AI Authentication, Security, and RBAC Services.
"""

import pytest
from induspredict.core.config import ROLE_ADMIN, ROLE_ENGINEER, ROLE_RESEARCHER, ROLE_VIEWER
from induspredict.core.security import (
    PERM_MACHINE_DELETE,
    PERM_MACHINE_EDIT,
    PERM_USER_MANAGEMENT,
    SessionManager,
    hash_password,
    verify_password,
)
from induspredict.db.models import AuditLog, User
from induspredict.services.auth_service import AuthService


def test_password_hashing():
    """Verify that bcrypt generates different salts and validates correctly."""
    pwd = "IndustrialSecure#2026"
    h1 = hash_password(pwd)
    h2 = hash_password(pwd)

    assert h1 != h2, "Salts must produce distinct hashes for identical passwords"
    assert verify_password(pwd, h1) is True
    assert verify_password(pwd, h2) is True
    assert verify_password("WrongPassword", h1) is False
    assert verify_password("", h1) is False


def test_default_user_authentication(test_engine):
    """Test login for all seeded default operator roles."""
    roles_credentials = [
        ("admin", "Admin@123", ROLE_ADMIN),
        ("engineer", "Engineer@123", ROLE_ENGINEER),
        ("researcher", "Researcher@123", ROLE_RESEARCHER),
        ("viewer", "Viewer@123", ROLE_VIEWER),
    ]

    for uname, pwd, expected_role in roles_credentials:
        session = AuthService.authenticate(uname, pwd)
        assert session is not None, f"Authentication failed for seeded user '{uname}'"
        assert session.username == uname
        assert session.role_name == expected_role
        assert session.is_active is True
        assert SessionManager.is_authenticated() is True

        AuthService.logout()
        assert SessionManager.is_authenticated() is False


def test_invalid_authentication(test_engine):
    """Test rejection of non-existent users and incorrect credentials."""
    assert AuthService.authenticate("non_existent_user", "Admin@123") is None
    assert AuthService.authenticate("admin", "WrongPassword999") is None
    assert AuthService.authenticate("", "") is None


def test_user_registration(test_engine):
    """Test user registration by Administrator with validation checks."""
    admin_session = AuthService.authenticate("admin", "Admin@123")
    assert admin_session is not None

    # Register new engineer
    res = AuthService.register_user(
        username="m_smith",
        email="m.smith@plant.com",
        password="PassWord123!",
        full_name="Marcus Smith",
        role_name=ROLE_ENGINEER,
        admin_session=admin_session,
    )
    assert res["success"] is True
    assert res["username"] == "m_smith"

    # Authenticate newly created user
    new_session = AuthService.authenticate("m_smith", "PassWord123!")
    assert new_session is not None
    assert new_session.role_name == ROLE_ENGINEER

    # Try duplicate registration
    res_dup = AuthService.register_user(
        username="m_smith",
        email="different@plant.com",
        password="PassWord123!",
        full_name="Duplicate Marcus",
        role_name=ROLE_ENGINEER,
        admin_session=admin_session,
    )
    assert res_dup["success"] is False
    assert "already registered" in res_dup["error"]

    AuthService.logout()


def test_rbac_permissions(test_engine):
    """Test role-based permissions matrix."""
    admin_session = AuthService.authenticate("admin", "Admin@123")
    assert admin_session.has_permission(PERM_USER_MANAGEMENT) is True
    assert admin_session.has_permission(PERM_MACHINE_DELETE) is True
    AuthService.logout()

    viewer_session = AuthService.authenticate("viewer", "Viewer@123")
    assert viewer_session.has_permission(PERM_USER_MANAGEMENT) is False
    assert viewer_session.has_permission(PERM_MACHINE_DELETE) is False
    AuthService.logout()


def test_audit_log_created_on_login(test_engine, db_session):
    """Test that authentication events are appended to AuditLog."""
    initial_count = db_session.query(AuditLog).filter(AuditLog.action == "LOGIN_SUCCESS").count()
    session = AuthService.authenticate("admin", "Admin@123")
    assert session is not None

    new_count = db_session.query(AuditLog).filter(AuditLog.action == "LOGIN_SUCCESS").count()
    assert new_count == initial_count + 1
    AuthService.logout()
