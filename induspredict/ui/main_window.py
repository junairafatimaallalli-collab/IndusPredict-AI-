"""
IndusPredict AI - Main Desktop Shell Window.
Coordinates sidebar navigation, role-based visibility, top industrial status bar,
and smooth QStackedWidget page transitions.
"""

from datetime import datetime
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.utils import utc_now
from induspredict.core.config import AppConfig
from induspredict.core.security import (
    PERM_MACHINE_EDIT,
    PERM_USER_MANAGEMENT,
    SessionManager,
    UserSession,
)
from induspredict.services.auth_service import AuthService
from induspredict.ui.theme import (
    BG_CARD,
    BG_DARK,
    BG_PANEL,
    BORDER_SUBTLE,
    COLOR_CRITICAL,
    COLOR_HEALTHY,
    COLOR_WARNING,
    CYAN_PRIMARY,
    INDUSTRIAL_QSS,
    TEAL_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from induspredict.ui.views.dashboard_view import DashboardView
from induspredict.ui.views.live_monitoring_view import LiveMonitoringView
from induspredict.ui.views.login_view import LoginView
from induspredict.ui.views.machine_view import MachineView
from induspredict.ui.views.prediction_view import PredictionView
from induspredict.ui.views.rul_view import RULPrognosticsView
from induspredict.ui.views.digital_twin_view import DigitalTwinView
from induspredict.ui.views.xai_view import XAIView
from induspredict.ui.views.maintenance_view import MaintenanceView
from induspredict.ui.views.reports_view import ReportsView


class PhasePlaceholderView(QWidget):
    """Clean placeholder view for upcoming development phases."""

    def __init__(self, phase_num: int, title: str, description: str, planned_features: List[str], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setProperty("class", "card")
        card.setMaximumWidth(700)
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(28, 24, 28, 24)
        c_layout.setSpacing(14)

        tag = QLabel(f"DEVELOPMENT ROADMAP • PHASE {phase_num}")
        tag.setStyleSheet(f"color: {CYAN_PRIMARY}; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        c_layout.addWidget(tag)

        h_lbl = QLabel(title)
        h_lbl.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {TEXT_PRIMARY};")
        c_layout.addWidget(h_lbl)

        d_lbl = QLabel(description)
        d_lbl.setWordWrap(True)
        d_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; line-height: 1.4;")
        c_layout.addWidget(d_lbl)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"border: none; border-bottom: 1px solid {BORDER_SUBTLE}; margin: 8px 0px;")
        c_layout.addWidget(div)

        feat_header = QLabel("Planned Engine Capabilities:")
        feat_header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: 600; font-size: 12px;")
        c_layout.addWidget(feat_header)

        for feat in planned_features:
            f_lbl = QLabel(f"• {feat}")
            f_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; margin-left: 8px;")
            c_layout.addWidget(f_lbl)

        layout.addWidget(card)


class MainWindow(QMainWindow):
    """Main Industrial Application Shell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IndusPredict AI – Industrial Predictive Maintenance")
        self.resize(1340, 860)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(INDUSTRIAL_QSS)

        self._init_shell()
        self._show_login_page()

    def _init_shell(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Industrial App Bar
        self.top_bar = QFrame()
        self.top_bar.setObjectName("topBar")
        self.top_bar.setFixedHeight(54)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)
        top_layout.setSpacing(12)

        # Title Brand
        title_box = QHBoxLayout()
        title_box.setSpacing(8)
        lbl_logo = QLabel("⚙️")
        lbl_logo.setStyleSheet("font-size: 20px;")
        lbl_title = QLabel("IndusPredict AI")
        lbl_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {CYAN_PRIMARY};")
        title_box.addWidget(lbl_logo)
        title_box.addWidget(lbl_title)
        top_layout.addLayout(title_box)

        # Safety Banner (ISO / CPS Safety)
        lbl_safety = QLabel("🔒 READ-ONLY INDUSTRIAL TELEMETRY INTERFACE")
        lbl_safety.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.08); color: {CYAN_PRIMARY}; border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 4px; padding: 3px 10px; font-size: 11px; font-weight: bold;"
        )
        top_layout.addWidget(lbl_safety)

        top_layout.addStretch()

        # Clock
        self.lbl_clock = QLabel("")
        self.lbl_clock.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-family: monospace;")
        top_layout.addWidget(self.lbl_clock)

        # User Profile Badge & Logout (Hidden on login screen)
        self.user_badge = QLabel("")
        self.user_badge.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 500;")
        top_layout.addWidget(self.user_badge)

        self.role_badge = QLabel("")
        self.role_badge.setStyleSheet(
            f"background-color: rgba(46, 204, 113, 0.15); color: {COLOR_HEALTHY}; border: 1px solid {COLOR_HEALTHY}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        top_layout.addWidget(self.role_badge)

        self.btn_logout = QPushButton("Logout")
        self.btn_logout.setProperty("class", "secondary-btn")
        self.btn_logout.setStyleSheet("padding: 4px 12px; font-size: 11px;")
        self.btn_logout.clicked.connect(self._handle_logout)
        top_layout.addWidget(self.btn_logout)

        root_layout.addWidget(self.top_bar)

        # 2. Main Content Area (Sidebar + Page Stack)
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Left Navigation Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sideBar")
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(12, 16, 12, 16)
        side_layout.setSpacing(6)

        nav_header = QLabel("NAVIGATION")
        nav_header.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-weight: bold; letter-spacing: 1px; padding-left: 10px; margin-bottom: 4px;")
        side_layout.addWidget(nav_header)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.nav_buttons: List[QPushButton] = []
        nav_items = [
            ("📊  Main Dashboard", 1),
            ("🏭  Machine Management", 2),
            ("📡  Live Sensor Monitoring", 3),
            ("🔮  Failure Prediction Engine", 4),
            ("⏳  RUL Prognostics (LSTM)", 5),
            ("🌐  Digital Twin Simulator", 6),
            ("💡  Explainable AI (SHAP)", 7),
            ("🛠️  Adaptive Planner & Drift", 8),
            ("📑  Reports & Analytics", 9),
        ]

        for text, page_idx in nav_items:
            btn = QPushButton(text)
            btn.setProperty("class", "nav-btn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, idx=page_idx: self._switch_page(idx))
            self.nav_group.addButton(btn, page_idx)
            self.nav_buttons.append(btn)
            side_layout.addWidget(btn)

        side_layout.addStretch()

        # Database Status in Sidebar Footer
        self.db_status = QLabel("● Database: SQLite (WAL Active)")
        self.db_status.setStyleSheet(f"color: {COLOR_HEALTHY}; font-size: 11px; padding-left: 8px;")
        side_layout.addWidget(self.db_status)

        body_layout.addWidget(self.sidebar)

        # Stacked Pages Widget
        self.page_stack = QStackedWidget()

        # Page 0: Login View
        self.login_view = LoginView()
        self.login_view.login_successful.connect(self._on_login_success)
        self.page_stack.addWidget(self.login_view)

        # Page 1: Main Dashboard
        self.dashboard_view = DashboardView()
        self.page_stack.addWidget(self.dashboard_view)

        # Page 2: Machine Management
        self.machine_view = MachineView()
        self.machine_view.machine_modified.connect(self.dashboard_view.refresh_data)
        self.page_stack.addWidget(self.machine_view)

        # Page 3: Live Sensor Monitoring (Phase 2)
        self.live_monitoring_view = LiveMonitoringView()
        self.machine_view.machine_modified.connect(self.live_monitoring_view.load_machines_list)
        self.page_stack.addWidget(self.live_monitoring_view)

        # Page 4: Failure Prediction Engine (Phase 3)
        self.prediction_view = PredictionView()
        self.machine_view.machine_modified.connect(self.prediction_view.load_machine_dropdown)
        self.page_stack.addWidget(self.prediction_view)

        # Page 5: RUL Prognostics (LSTM - Phase 4)
        self.rul_view = RULPrognosticsView()
        self.page_stack.addWidget(self.rul_view)

        # Page 6: Digital Twin Simulator (Phase 5)
        self.digital_twin_view = DigitalTwinView()
        self.page_stack.addWidget(self.digital_twin_view)

        # Page 7: Explainable AI (SHAP - Phase 6)
        self.xai_view = XAIView()
        self.page_stack.addWidget(self.xai_view)

        # Page 8: Adaptive Maintenance Planner & Drift Monitor (Phase 7)
        self.maintenance_view = MaintenanceView()
        self.page_stack.addWidget(self.maintenance_view)

        # Page 9: Reports & Analytics (Phase 8)
        self.reports_view = ReportsView()
        self.page_stack.addWidget(self.reports_view)

        body_layout.addWidget(self.page_stack)
        root_layout.addWidget(body)

        # Setup Clock Timer
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start()
        self._update_clock()

    def _update_clock(self) -> None:
        self.lbl_clock.setText(utc_now().strftime("UTC %Y-%m-%d %H:%M:%S"))

    def _show_login_page(self) -> None:
        """Display login screen and hide authenticated navigation elements."""
        self.sidebar.hide()
        self.user_badge.hide()
        self.role_badge.hide()
        self.btn_logout.hide()
        self.page_stack.setCurrentIndex(0)

    def _on_login_success(self, session: UserSession) -> None:
        """Invoked when user provides valid credentials."""
        self.user_badge.setText(f"👤 {session.full_name} ({session.username})")
        self.role_badge.setText(session.role_name.upper())

        # Show authenticated top bar and sidebar
        self.user_badge.show()
        self.role_badge.show()
        self.btn_logout.show()
        self.sidebar.show()

        # Switch to Main Dashboard (Page 1)
        self._switch_page(1)
        self.dashboard_view.refresh_data()
        self.machine_view.load_machines()

    def _switch_page(self, index: int) -> None:
        """Switch stacked page and update active navigation button state."""
        self.page_stack.setCurrentIndex(index)
        for btn in self.nav_buttons:
            btn_idx = self.nav_group.id(btn)
            btn.setChecked(btn_idx == index)

        # Trigger refresh on destination view if applicable
        if index == 1:
            self.dashboard_view.refresh_data()
        elif index == 2:
            self.machine_view.load_machines()
        elif index == 3:
            self.live_monitoring_view.load_machines_list()
        elif index == 4:
            self.prediction_view.load_machine_dropdown()
            self.prediction_view.load_benchmarks()
            self.prediction_view.load_history()
        elif index == 5:
            self.rul_view.update_trajectory_chart()
            self.rul_view.load_history()

    def _handle_logout(self) -> None:
        """Terminate user session and return to login screen."""
        reply = QMessageBox.question(
            self,
            "Confirm Logout",
            "Are you sure you want to end the active session?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            AuthService.logout()
            self._show_login_page()
