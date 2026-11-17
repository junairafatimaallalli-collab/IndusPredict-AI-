"""
Automated Headless GUI Tests for Reports & Analytics View (Phase 8).
"""

from pathlib import Path
import tempfile
import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.reports_view import ReportsView


def test_reports_view_initialization(qapp, test_engine):
    """Test ReportsView widget creation, controls, tables, and KPI cards."""
    AuthService.authenticate("admin", "Admin@123")

    view = ReportsView()
    view.show()

    # Check report types
    assert view.cb_report_type.count() == 5
    assert view.cb_machine.count() >= 6  # All + 5 machines

    # Check KPI cards
    assert len(view.kpi_cards) == 4

    # Check table preview has columns and rows
    assert view.tbl_preview.columnCount() > 0
    assert view.tbl_preview.rowCount() > 0

    # Test switching report type
    view.cb_report_type.setCurrentIndex(1)
    assert view.tbl_preview.columnCount() > 0

    view.cb_report_type.setCurrentIndex(2)
    assert view.tbl_preview.columnCount() > 0

    AuthService.logout()


def test_main_window_phase8_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 9 (Reports & Analytics)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 9 (Reports & Analytics)
    window._switch_page(9)
    assert window.page_stack.currentIndex() == 9
    assert isinstance(window.page_stack.currentWidget(), ReportsView)

    AuthService.logout()
