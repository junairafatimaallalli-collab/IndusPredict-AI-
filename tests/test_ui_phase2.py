"""
Automated Headless GUI Tests for Live Sensor Monitoring (Phase 2).
"""

import pytest
from induspredict.services.auth_service import AuthService
from induspredict.ui.main_window import MainWindow
from induspredict.ui.views.live_monitoring_view import LiveMonitoringView


def test_live_monitoring_view(qapp, test_engine):
    """Test LiveMonitoringView rendering, simulator toggle, and fault injection."""
    AuthService.authenticate("admin", "Admin@123")

    view = LiveMonitoringView()
    view.show()

    assert view.cb_machines.count() >= 5
    assert view.card_temp is not None
    assert view.card_vib is not None
    assert view.card_press is not None
    assert view.chart_live is not None

    # Test toggling simulator
    assert view.sim_worker.isRunning() is False
    view._toggle_simulation()
    assert view.sim_worker.isRunning() is True
    assert "PHYSICS SIMULATOR" in view.lbl_source_badge.text()

    # Step simulator manually
    frame = view.simulator.generate_step()
    assert frame is not None
    view._on_telemetry_frame(frame)

    # Verify card reading updated
    assert "--" not in view.card_temp.lbl_val.text()

    # Test fault mode change
    view.cb_fault_mode.setCurrentText(view.simulator.MODE_THERMAL_OVERRUN)
    assert view.simulator.mode == view.simulator.MODE_THERMAL_OVERRUN

    # Test stop
    view._toggle_simulation()
    assert view.sim_worker.isRunning() is False

    AuthService.logout()


def test_main_window_phase2_navigation(qapp, test_engine):
    """Test navigating to Page 3 (Live Sensor Monitoring) in MainWindow."""
    session = AuthService.authenticate("admin", "Admin@123")
    window = MainWindow()
    window.show()
    window._on_login_success(session)

    # Switch to Page 3 (Live Sensor Monitoring)
    window._switch_page(3)
    assert window.page_stack.currentIndex() == 3
    assert isinstance(window.page_stack.currentWidget(), LiveMonitoringView)

    AuthService.logout()
