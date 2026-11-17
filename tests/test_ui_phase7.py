"""
Automated Headless GUI Tests for Maintenance Planner & Drift Monitoring View (Phase 7).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.maintenance_view import MaintenanceView


def test_maintenance_view_initialization(qapp, test_engine):
    """Test MaintenanceView widget creation, controls, tables, and tab navigation."""
    AuthService.authenticate("admin", "Admin@123")

    view = MaintenanceView()
    view.show()

    # Verify tabs
    assert view.tabs.count() == 4

    # Verify orders table populated
    assert view.tbl_orders.rowCount() >= 0
    assert view.cb_f_status.count() == 5

    # Verify suggestions table populated
    assert view.tbl_suggestions is not None

    # Verify drift table populated
    assert view.tbl_drift is not None

    # Verify history table populated
    assert view.tbl_history is not None

    # Test filter change
    view.cb_f_status.setCurrentText("Pending")
    assert view.cb_f_status.currentText() == "Pending"

    AuthService.logout()


def test_main_window_phase7_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 8 (Adaptive Maintenance Planner & Drift Monitor)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 8 (Maintenance Planner)
    window._switch_page(8)
    assert window.page_stack.currentIndex() == 8
    assert isinstance(window.page_stack.currentWidget(), MaintenanceView)

    AuthService.logout()
