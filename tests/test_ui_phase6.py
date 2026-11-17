"""
Automated Headless GUI Tests for Explainable AI (SHAP) View (Phase 6).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.xai_view import XAIView


def test_xai_view_initialization(qapp, test_engine):
    """Test XAIView widget creation, controls, tables, and charts."""
    AuthService.authenticate("admin", "Admin@123")

    view = XAIView()
    view.show()

    # Verify model dropdown
    assert view.cb_models.count() == 3
    assert view.tabs.count() == 2

    # Verify narrative text populated
    assert len(view.txt_narrative.toPlainText()) > 0
    assert "Driver" in view.txt_narrative.toPlainText() or "Nominal" in view.txt_narrative.toPlainText()

    # Verify normal vs abnormal table populated
    assert view.tbl_comparison.rowCount() > 0

    # Verify global importance table populated
    assert view.tbl_global.rowCount() == 10

    # Test presets
    view._preset_thermal()
    narrative_lower = view.txt_narrative.toPlainText().lower()
    assert "temperature" in narrative_lower or "thermal" in narrative_lower or "temp" in narrative_lower

    view._preset_strain()
    assert "Driver" in view.txt_narrative.toPlainText()

    # Test switching model to XGBoost
    view.cb_models.setCurrentText("XGBoost")
    assert view.current_model == "XGBoost"

    AuthService.logout()


def test_main_window_phase6_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 7 (Explainable AI - SHAP)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 7 (Explainable AI)
    window._switch_page(7)
    assert window.page_stack.currentIndex() == 7
    assert isinstance(window.page_stack.currentWidget(), XAIView)

    AuthService.logout()
