"""
Automated Headless GUI Tests for Failure Prediction View (Phase 3).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.prediction_view import PredictionView


def test_prediction_view_initialization(qapp, test_engine):
    """Test PredictionView widget creation, controls, and benchmark tables."""
    AuthService.authenticate("admin", "Admin@123")

    view = PredictionView()
    view.show()

    assert view.cb_model.count() == 4
    assert view.cb_target_machine.count() >= 5
    assert view.btn_predict is not None
    assert view.chart_contributions is not None

    # Check benchmark comparison table populated with 4 models
    assert view.tbl_benchmarks.rowCount() == 4

    # Check prediction history populated
    assert view.tbl_history.rowCount() > 0

    # Test preset buttons
    view._preset_thermal_overload()
    assert view.spin_speed.value() == 1320.0

    view._preset_overstrain()
    assert view.spin_torque.value() == 68.0

    view._preset_nominal()
    assert view.spin_air_t.value() == 25.0

    # Test switching algorithms
    view.cb_model.setCurrentText("XGBoost")
    view.run_prediction()
    assert "%" in view.lbl_prob_val.text()

    view.cb_model.setCurrentText("LightGBM")
    view.run_prediction()
    assert "%" in view.lbl_prob_val.text()

    AuthService.logout()


def test_main_window_phase3_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 4 (Failure Prediction Engine)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 4 (Failure Prediction)
    window._switch_page(4)
    assert window.page_stack.currentIndex() == 4
    assert isinstance(window.page_stack.currentWidget(), PredictionView)

    AuthService.logout()
