"""
Automated Headless GUI Tests for RUL Prognostics View (Phase 4).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.rul_view import RULPrognosticsView


def test_rul_prognostics_view_initialization(qapp, test_engine):
    """Test RULPrognosticsView widget creation, controls, tables, and charts."""
    AuthService.authenticate("admin", "Admin@123")

    view = RULPrognosticsView()
    view.show()

    # Check dropdown has units
    assert view.cb_units.count() > 0
    assert view.tabs.count() == 3

    # Check KPI cards
    assert "Cycles" in view.lbl_rul_val.text()
    assert "%" in view.lbl_deg_val.text()

    # Check benchmark table has entries (LSTM vs Baseline)
    assert view.tbl_comparison.rowCount() >= 2

    # Check history table exists
    assert view.tbl_history is not None

    # Test changing unit
    initial_unit = view.current_unit_id
    if view.cb_units.count() > 1:
        view.cb_units.setCurrentIndex(1)
        assert view.current_unit_id != initial_unit

    AuthService.logout()


def test_main_window_phase4_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 5 (RUL Prognostics Engine)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 5 (RUL Prognostics)
    window._switch_page(5)
    assert window.page_stack.currentIndex() == 5
    assert isinstance(window.page_stack.currentWidget(), RULPrognosticsView)

    AuthService.logout()
