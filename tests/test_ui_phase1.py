"""
Automated Headless GUI Tests for IndusPredict AI (Phase 1).
Verifies Qt widget rendering, interactions, signal bindings, and page navigation.
"""

import pytest
from PySide6.QtCore import Qt
from induspredict.core.security import SessionManager
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.dashboard_view import DashboardView
from induspredict.ui.views.login_view import LoginView
from induspredict.ui.views.machine_view import MachineView


def test_login_view(qapp, test_engine):
    """Test LoginView widget input, credentials setting, and signal emission."""
    login_widget = LoginView()
    login_widget.show()
    assert login_widget.txt_username.text() == "admin"
    assert login_widget.txt_password.text() == "Admin@123"

    received_sessions = []
    login_widget.login_successful.connect(lambda s: received_sessions.append(s))

    # Trigger login
    login_widget._handle_login()
    assert len(received_sessions) == 1
    session = received_sessions[0]
    assert session.username == "admin"
    assert session.role_name == "Administrator"

    # Test invalid login error banner
    login_widget.txt_password.setText("WrongPassword!")
    login_widget._handle_login()
    assert not login_widget.lbl_error.isHidden()
    assert "failed" in login_widget.lbl_error.text().lower()

    AuthService.logout()


def test_dashboard_view(qapp, test_engine):
    """Test DashboardView initialization, dynamic card values, and chart rendering."""
    dashboard = DashboardView()
    dashboard.show()
    assert dashboard.card_total.lbl_value.text() != "0"
    assert int(dashboard.card_total.lbl_value.text()) >= 5

    # Check predictions table populated
    assert dashboard.tbl_predictions.rowCount() > 0
    # Check charts drawn
    assert dashboard.chart_trends.canvas is not None
    assert dashboard.chart_risk.canvas is not None

    # Test machine filter change
    dashboard.cb_machine_filter.setCurrentIndex(1)
    dashboard.refresh_data()


def test_machine_view(qapp, test_engine):
    """Test MachineView table rows, selection change, and search filtering."""
    # Ensure active session so RBAC permissions pass
    AuthService.authenticate("admin", "Admin@123")

    machine_view = MachineView()
    machine_view.show()
    assert machine_view.table.rowCount() >= 5

    # Select first machine
    machine_view.table.selectRow(0)
    assert machine_view.btn_edit.isEnabled() is True
    assert machine_view.btn_delete.isEnabled() is True
    assert machine_view.tbl_components.rowCount() > 0
    assert machine_view.tbl_sensors.rowCount() > 0

    # Test search filtering
    machine_view.txt_search.setText("KUKA")
    assert machine_view.table.rowCount() == 1
    assert "KUKA" in machine_view.table.item(0, 2).text()

    machine_view.txt_search.setText("")
    assert machine_view.table.rowCount() >= 5

    AuthService.logout()


def test_main_window_navigation(qapp, test_engine):
    """Test MainWindow page transitions, authentication flow, and logout."""
    window = MainWindow()
    window.show()

    # Initial state: on Login page (index 0), sidebar hidden
    assert window.page_stack.currentIndex() == 0
    assert window.sidebar.isHidden() is True

    # Simulate successful login
    session = AuthService.authenticate("admin", "Admin@123")
    window._on_login_success(session)

    # State post-login: on Dashboard page (index 1), sidebar visible
    assert window.page_stack.currentIndex() == 1
    assert window.sidebar.isHidden() is False
    assert "admin" in window.user_badge.text()
    assert "ADMINISTRATOR" in window.role_badge.text()

    # Navigate to Machine Management (index 2)
    window._switch_page(2)
    assert window.page_stack.currentIndex() == 2

    # Navigate to Live Sensors (index 3 - Roadmap phase 2)
    window._switch_page(3)
    assert window.page_stack.currentIndex() == 3

    # Direct logout call
    AuthService.logout()
    window._show_login_page()
    assert window.page_stack.currentIndex() == 0
    assert window.sidebar.isHidden() is True
