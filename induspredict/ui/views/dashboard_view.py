"""
IndusPredict AI - Main Operations Dashboard View.
Renders real-time aggregated KPIs from SQLite, interactive Matplotlib charts,
recent failure prediction logs, active critical alerts, and asset filters.
"""

from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from induspredict.services.dashboard_service import DashboardService
from induspredict.services.machine_service import MachineService
from induspredict.ui.theme import (
    BG_CARD,
    BG_PANEL,
    BORDER_SUBTLE,
    COLOR_CRITICAL,
    COLOR_HEALTHY,
    COLOR_INFO,
    COLOR_WARNING,
    CYAN_PRIMARY,
    TEAL_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from induspredict.ui.widgets.chart_view import ChartCard
from induspredict.ui.widgets.stat_card import StatCard


class DashboardView(QWidget):
    """Dynamic industrial dashboard visualizing real-time fleet health."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_machine_id: Optional[int] = None
        self._init_ui()
        self._setup_auto_refresh()
        self.refresh_data()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(16)

        # 1. Header & Global Filters Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Fleet Operations & Asset Health Dashboard")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Real-time telemetry aggregation, failure classification, and prognostic risk distribution")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()

        # Machine Filter Dropdown
        filter_box = QHBoxLayout()
        filter_box.setSpacing(8)
        lbl_filter = QLabel("Target Asset:")
        lbl_filter.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 600;")
        self.cb_machine_filter = QComboBox()
        self.cb_machine_filter.setMinimumWidth(220)
        self.cb_machine_filter.addItem("All Registered Equipment", None)
        self.cb_machine_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_box.addWidget(lbl_filter)
        filter_box.addWidget(self.cb_machine_filter)
        header_row.addLayout(filter_box)

        # Refresh Button
        self.btn_refresh = QPushButton("⟳ Refresh Telemetry")
        self.btn_refresh.setProperty("class", "secondary-btn")
        self.btn_refresh.clicked.connect(self.refresh_data)
        header_row.addWidget(self.btn_refresh)

        main_layout.addLayout(header_row)

        # 2. Scrollable Content Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # 3. Dynamic KPI Cards Grid (6 actual database counters)
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(12)

        self.card_total = StatCard("Total Machines", "0", "🏭", "Active factory assets", CYAN_PRIMARY)
        self.card_healthy = StatCard("Healthy Machines", "0", "✅", "Operating within nominal envelopes", COLOR_HEALTHY)
        self.card_obs = StatCard("Under Observation", "0", "⚠️", "Mild sensor variance detected", COLOR_WARNING)
        self.card_risk = StatCard("High-Risk Machines", "0", "🚨", "Degradation threshold crossed", COLOR_CRITICAL)
        self.card_alerts = StatCard("Critical Alerts", "0", "🔔", "Active unacknowledged triggers", COLOR_CRITICAL)
        self.card_tasks = StatCard("Pending Work Orders", "0", "📋", "Maintenance actions queued", COLOR_INFO)

        kpi_grid.addWidget(self.card_total, 0, 0)
        kpi_grid.addWidget(self.card_healthy, 0, 1)
        kpi_grid.addWidget(self.card_obs, 0, 2)
        kpi_grid.addWidget(self.card_risk, 1, 0)
        kpi_grid.addWidget(self.card_alerts, 1, 1)
        kpi_grid.addWidget(self.card_tasks, 1, 2)

        content_layout.addLayout(kpi_grid)

        # 4. Interactive Charts Row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        self.chart_trends = ChartCard("Machine Health Trajectory & Anomaly Frequency (Last 12 Hrs)")
        self.chart_risk = ChartCard("Fleet Failure Risk Stratification")

        charts_row.addWidget(self.chart_trends, 2)
        charts_row.addWidget(self.chart_risk, 1)
        content_layout.addLayout(charts_row)

        # 5. Bottom Tables Row: Recent Predictions & Active Alerts
        tables_row = QHBoxLayout()
        tables_row.setSpacing(12)

        # Recent Predictions Card
        pred_card = QFrame()
        pred_card.setProperty("class", "card")
        pred_layout = QVBoxLayout(pred_card)
        pred_layout.setContentsMargins(12, 12, 12, 12)
        pred_layout.setSpacing(8)

        lbl_pred_title = QLabel("LATEST MACHINE FAILURE PREDICTIONS")
        lbl_pred_title.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {CYAN_PRIMARY};")
        pred_layout.addWidget(lbl_pred_title)

        self.tbl_predictions = QTableWidget()
        self.tbl_predictions.setColumnCount(6)
        self.tbl_predictions.setHorizontalHeaderLabels([
            "Time", "Asset Code", "Model", "Failure Prob", "Risk Level", "Predicted Mode"
        ])
        self.tbl_predictions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_predictions.verticalHeader().setVisible(False)
        self.tbl_predictions.setMinimumHeight(200)
        pred_layout.addWidget(self.tbl_predictions)

        # Recent Alerts Card
        alert_card = QFrame()
        alert_card.setProperty("class", "card")
        alert_layout = QVBoxLayout(alert_card)
        alert_layout.setContentsMargins(12, 12, 12, 12)
        alert_layout.setSpacing(8)

        lbl_alert_title = QLabel("ACTIVE CRITICAL & ANOMALY ALERTS")
        lbl_alert_title.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLOR_CRITICAL};")
        alert_layout.addWidget(lbl_alert_title)

        self.tbl_alerts = QTableWidget()
        self.tbl_alerts.setColumnCount(5)
        self.tbl_alerts.setHorizontalHeaderLabels([
            "Timestamp", "Asset", "Sensor", "Severity", "Description"
        ])
        self.tbl_alerts.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_alerts.verticalHeader().setVisible(False)
        self.tbl_alerts.setMinimumHeight(200)
        alert_layout.addWidget(self.tbl_alerts)

        tables_row.addWidget(pred_card, 1)
        tables_row.addWidget(alert_card, 1)
        content_layout.addLayout(tables_row)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def _setup_auto_refresh(self) -> None:
        """Periodic auto-refresh every 15 seconds to reflect incoming telemetry."""
        self.timer = QTimer(self)
        self.timer.setInterval(15000)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start()

    def _populate_filter_dropdown(self) -> None:
        """Populate the filter dropdown with actual machine codes."""
        current_data = self.cb_machine_filter.currentData()
        self.cb_machine_filter.blockSignals(True)
        self.cb_machine_filter.clear()
        self.cb_machine_filter.addItem("All Registered Equipment", None)

        machines = MachineService.get_all_machines()
        for m in machines:
            self.cb_machine_filter.addItem(f"{m['code']} - {m['name'][:24]}...", m["id"])

        # Restore previous selection if still valid
        idx = self.cb_machine_filter.findData(current_data)
        if idx >= 0:
            self.cb_machine_filter.setCurrentIndex(idx)
        else:
            self.cb_machine_filter.setCurrentIndex(0)
        self.cb_machine_filter.blockSignals(False)

    def _on_filter_changed(self, index: int) -> None:
        self.selected_machine_id = self.cb_machine_filter.currentData()
        self.refresh_data()

    def refresh_data(self) -> None:
        """Fetch fresh dynamic analytics from SQLite and update all UI elements."""
        self._populate_filter_dropdown()

        # 1. Update KPI Cards
        kpis = DashboardService.get_summary_kpis(self.selected_machine_id)
        self.card_total.update_value(kpis["total_machines"])
        self.card_healthy.update_value(kpis["healthy_machines"])
        self.card_obs.update_value(kpis["observation_machines"])
        self.card_risk.update_value(kpis["high_risk_machines"])
        self.card_alerts.update_value(kpis["critical_alerts"])
        self.card_tasks.update_value(kpis["pending_tasks"])

        # 2. Update Risk Distribution Chart
        risk_dist = DashboardService.get_failure_risk_distribution(self.selected_machine_id)
        self.chart_risk.plot_risk_distribution(risk_dist)

        # 3. Update Health Trend Chart
        trends = DashboardService.get_health_trends(self.selected_machine_id, hours=12)
        self.chart_trends.plot_health_trends(
            timestamps=trends["timestamps"],
            values=trends["values"],
            anomalies=trends["anomalies"],
        )

        # 4. Update Recent Predictions Table
        predictions = DashboardService.get_recent_predictions(limit=6, machine_id_filter=self.selected_machine_id)
        self.tbl_predictions.setRowCount(len(predictions))
        for row, p in enumerate(predictions):
            self.tbl_predictions.setItem(row, 0, QTableWidgetItem(str(p["timestamp"])))
            self.tbl_predictions.setItem(row, 1, QTableWidgetItem(str(p["machine_code"])))
            self.tbl_predictions.setItem(row, 2, QTableWidgetItem(str(p["model_name"][:18])))
            self.tbl_predictions.setItem(row, 3, QTableWidgetItem(f"{p['failure_prob']}%"))

            # Risk Category Badge
            cat_item = QTableWidgetItem(str(p["risk_category"]))
            if p["risk_category"] == "Normal":
                cat_item.setForeground(Qt.green)
            elif p["risk_category"] == "Warning":
                cat_item.setForeground(Qt.yellow)
            else:
                cat_item.setForeground(Qt.red)
            self.tbl_predictions.setItem(row, 4, cat_item)

            self.tbl_predictions.setItem(row, 5, QTableWidgetItem(str(p["failure_mode"])))

        # 5. Update Recent Alerts Table
        alerts = DashboardService.get_recent_alerts(limit=6, machine_id_filter=self.selected_machine_id)
        self.tbl_alerts.setRowCount(len(alerts))
        for row, a in enumerate(alerts):
            self.tbl_alerts.setItem(row, 0, QTableWidgetItem(str(a["timestamp"])))
            self.tbl_alerts.setItem(row, 1, QTableWidgetItem(str(a["machine_code"])))
            self.tbl_alerts.setItem(row, 2, QTableWidgetItem(str(a["sensor_name"])))

            sev_item = QTableWidgetItem(str(a["severity"]))
            if a["severity"] == "Critical":
                sev_item.setForeground(Qt.red)
            else:
                sev_item.setForeground(Qt.yellow)
            self.tbl_alerts.setItem(row, 3, sev_item)

            self.tbl_alerts.setItem(row, 4, QTableWidgetItem(str(a["description"])))
