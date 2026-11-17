"""
Automated Headless GUI Tests for Digital Twin Simulator View (Phase 5).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.digital_twin_view import DigitalTwinView


def test_digital_twin_view_initialization(qapp, test_engine):
    """Test DigitalTwinView widget creation, controls, preset simulations, and tables."""
    AuthService.authenticate("admin", "Admin@123")

    view = DigitalTwinView()
    view.show()

    # Verify machine dropdown
    assert view.cb_machines.count() >= 5
    assert view.tabs.count() == 2

    # Verify spinboxes
    assert view.spin_air_t.value() == 25.0
    assert view.spin_speed.value() == 1500.0

    # Verify KPI cards
    assert "Baseline:" in view.lbl_base_health.text()
    assert "Simulated:" in view.lbl_sim_health.text()
    assert "Baseline:" in view.lbl_base_risk.text()

    # Test presets
    view._preset_thermal()
    assert view.spin_air_t.value() == 35.0
    assert view.spin_proc_t.value() == 48.0

    view._preset_heavy_load()
    assert view.spin_torque.value() == 68.0

    view._preset_baseline()
    assert view.spin_air_t.value() == 25.0

    # Verify chart exists
    assert view.chart_comparison is not None

    # Verify scenario table exists
    assert view.tbl_scenarios is not None

    AuthService.logout()


def test_main_window_phase5_navigation(qapp, test_engine):
    """Test MainWindow navigation to Page 6 (Digital Twin Simulator)."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 6 (Digital Twin Simulator)
    window._switch_page(6)
    assert window.page_stack.currentIndex() == 6
    assert isinstance(window.page_stack.currentWidget(), DigitalTwinView)

    AuthService.logout()
