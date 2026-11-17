"""
IndusPredict AI - Adaptive Maintenance Planner & Drift Monitoring View (Page 9).
Manages maintenance work orders, automated AI suggestions, lifecycle approvals,
engineer assignments, downtime estimates, and ADWIN/Page-Hinkley concept drift alerts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.logger import get_logger
from induspredict.services.machine_service import MachineService
from induspredict.services.maintenance_service import MaintenanceService
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

logger = get_logger("ui.maintenance")


class WorkOrderDialog(QDialog):
    """Modal dialog for creating a new maintenance work order."""

    def __init__(self, parent=None, initial_data: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Create Adaptive Maintenance Work Order")
        self.setMinimumWidth(460)
        self.initial_data = initial_data or {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)

        # Machine selector
        self.cb_machine = QComboBox()
        machines = MachineService.get_all_machines()
        for m in machines:
            self.cb_machine.addItem(f"{m['code']} - {m['name']}", m["code"])
        if "machine_code" in self.initial_data:
            idx = self.cb_machine.findData(self.initial_data["machine_code"])
            if idx >= 0:
                self.cb_machine.setCurrentIndex(idx)
        form.addRow("Target Asset:", self.cb_machine)

        # Title
        self.txt_title = QLineEdit()
        self.txt_title.setText(self.initial_data.get("title", ""))
        self.txt_title.setPlaceholderText("e.g. Compressor Bearing Replacement")
        form.addRow("Task Title:", self.txt_title)

        # Priority
        self.cb_priority = QComboBox()
        self.cb_priority.addItems(["Low", "Medium", "High", "Emergency"])
        if "priority" in self.initial_data:
            self.cb_priority.setCurrentText(self.initial_data["priority"])
        else:
            self.cb_priority.setCurrentText("Medium")
        form.addRow("Priority:", self.cb_priority)

        # Downtime estimate
        self.spin_downtime = QDoubleSpinBox()
        self.spin_downtime.setRange(0.25, 48.0)
        self.spin_downtime.setValue(self.initial_data.get("suggested_downtime_hours", 2.0))
        self.spin_downtime.setSingleStep(0.5)
        self.spin_downtime.setSuffix(" hrs")
        form.addRow("Estimated Downtime:", self.spin_downtime)

        # Description
        self.txt_desc = QTextEdit()
        self.txt_desc.setFixedHeight(80)
        self.txt_desc.setText(self.initial_data.get("description", ""))
        form.addRow("Scope / Description:", self.txt_desc)

        layout.addLayout(form)

        # Button box
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Create Work Order")
        btn_save.setProperty("class", "btn-primary")
        btn_save.clicked.connect(self._on_save)
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_save)
        layout.addLayout(btn_box)

    def _on_save(self) -> None:
        if not self.txt_title.text().strip():
            QMessageBox.warning(self, "Validation Error", "Task title is required.")
            return
        self.accept()

    def get_data(self) -> Dict[str, Any]:
        return {
            "machine_code": self.cb_machine.currentData(),
            "title": self.txt_title.text().strip(),
            "priority": self.cb_priority.currentText(),
            "estimated_downtime_hours": self.spin_downtime.value(),
            "description": self.txt_desc.toPlainText().strip() or "Standard scheduled maintenance.",
        }


class MaintenanceView(QWidget):
    """Page 9: Adaptive Maintenance Planner & Drift Monitoring View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.load_work_orders()
        self.load_ai_suggestions()
        self.load_drift_events()
        self.load_history()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Adaptive Maintenance Planner & Concept Drift Monitor")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Risk-prioritized work orders, lifecycle approvals, and ADWIN/Page-Hinkley streaming drift detection")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()

        btn_new_wo = QPushButton("➕ Create Work Order")
        btn_new_wo.setProperty("class", "btn-primary")
        btn_new_wo.clicked.connect(self._create_new_work_order)
        header_row.addWidget(btn_new_wo)

        main_layout.addLayout(header_row)

        # 2. Main Tabs
        self.tabs = QTabWidget()

        # TAB 1: Work Orders Kanban / List
        tab_wo = QWidget()
        wo_layout = QVBoxLayout(tab_wo)
        wo_layout.setContentsMargins(12, 12, 12, 12)
        wo_layout.setSpacing(12)

        # Filter row
        f_row = QHBoxLayout()
        f_row.setSpacing(10)
        lbl_f_status = QLabel("Status:")
        self.cb_f_status = QComboBox()
        self.cb_f_status.addItems(["All", "Pending", "Approved", "In Progress", "Completed"])
        self.cb_f_status.currentTextChanged.connect(lambda _: self.load_work_orders())
        f_row.addWidget(lbl_f_status)
        f_row.addWidget(self.cb_f_status)

        lbl_f_priority = QLabel("Priority:")
        self.cb_f_priority = QComboBox()
        self.cb_f_priority.addItems(["All", "Emergency", "High", "Medium", "Low"])
        self.cb_f_priority.currentTextChanged.connect(lambda _: self.load_work_orders())
        f_row.addWidget(lbl_f_priority)
        f_row.addWidget(self.cb_f_priority)

        f_row.addStretch()

        self.btn_approve = QPushButton("✓ Approve Selected")
        self.btn_approve.clicked.connect(self._approve_selected)
        f_row.addWidget(self.btn_approve)

        self.btn_complete = QPushButton("✔ Mark Completed")
        self.btn_complete.clicked.connect(self._complete_selected)
        f_row.addWidget(self.btn_complete)

        wo_layout.addLayout(f_row)

        self.tbl_orders = QTableWidget()
        self.tbl_orders.setColumnCount(8)
        self.tbl_orders.setHorizontalHeaderLabels([
            "ID", "WO Code", "Asset", "Task Title", "Priority", "Status", "Est. Downtime", "Scheduled Date"
        ])
        self.tbl_orders.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_orders.verticalHeader().setVisible(False)
        self.tbl_orders.setSelectionBehavior(QTableWidget.SelectRows)
        wo_layout.addWidget(self.tbl_orders)

        self.tabs.addTab(tab_wo, "🛠️ Active Work Orders")

        # TAB 2: AI Suggested Tasks
        tab_sugg = QWidget()
        sugg_layout = QVBoxLayout(tab_sugg)
        sugg_layout.setContentsMargins(16, 16, 16, 16)
        sugg_layout.setSpacing(12)

        lbl_sugg_hdr = QLabel("Automated Machine Risk-Prioritized Suggestions")
        lbl_sugg_hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        sugg_layout.addWidget(lbl_sugg_hdr)

        lbl_sugg_desc = QLabel(
            "IndusPredict AI periodically scans predictive failure probabilities, RUL horizons, and component lifetime hours "
            "to formulate proactive maintenance tasks before physical breakdowns occur."
        )
        lbl_sugg_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        sugg_layout.addWidget(lbl_sugg_desc)

        self.tbl_suggestions = QTableWidget()
        self.tbl_suggestions.setColumnCount(6)
        self.tbl_suggestions.setHorizontalHeaderLabels([
            "Target Asset", "Recommended Task", "Priority", "Est. Downtime", "Trigger Source", "Action"
        ])
        self.tbl_suggestions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_suggestions.verticalHeader().setVisible(False)
        sugg_layout.addWidget(self.tbl_suggestions)

        self.tabs.addTab(tab_sugg, "🤖 AI Adaptive Suggestions")

        # TAB 3: Model Concept Drift Monitoring
        tab_drift = QWidget()
        drift_layout = QVBoxLayout(tab_drift)
        drift_layout.setContentsMargins(16, 16, 16, 16)
        drift_layout.setSpacing(12)

        lbl_drift_hdr = QLabel("Statistical Concept Drift & Sensor Distribution Shifts")
        lbl_drift_hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        drift_layout.addWidget(lbl_drift_hdr)

        # Policy Alert Box
        p_frame = QFrame()
        p_frame.setStyleSheet(
            f"background-color: rgba(255, 122, 0, 0.08); border: 1px solid rgba(255, 122, 0, 0.25); border-radius: 6px; padding: 6px;"
        )
        p_lay = QHBoxLayout(p_frame)
        p_lay.setContentsMargins(10, 6, 10, 6)
        lbl_policy = QLabel(
            "⚠️ STRICT DEPLOYMENT POLICY: Replacement or retrained models are NEVER automatically deployed into production. "
            "Any distribution shift flagged by ADWIN or Page-Hinkley requires engineering verification and manual sign-off."
        )
        lbl_policy.setWordWrap(True)
        lbl_policy.setStyleSheet("color: #FFB300; font-size: 11px;")
        p_lay.addWidget(lbl_policy)
        drift_layout.addWidget(p_frame)

        self.tbl_drift = QTableWidget()
        self.tbl_drift.setColumnCount(7)
        self.tbl_drift.setHorizontalHeaderLabels([
            "ID", "Model / Telemetry", "Detector", "Statistic", "Threshold", "Status", "Recommendation"
        ])
        self.tbl_drift.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_drift.verticalHeader().setVisible(False)
        drift_layout.addWidget(self.tbl_drift)

        self.tabs.addTab(tab_drift, "📡 Drift & Model Review")

        # TAB 4: Archival Maintenance History
        tab_hist = QWidget()
        hist_layout = QVBoxLayout(tab_hist)
        hist_layout.setContentsMargins(16, 16, 16, 16)
        hist_layout.setSpacing(12)

        lbl_hist_hdr = QLabel("Archived Maintenance & Service Records")
        lbl_hist_hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        hist_layout.addWidget(lbl_hist_hdr)

        self.tbl_history = QTableWidget()
        self.tbl_history.setColumnCount(6)
        self.tbl_history.setHorizontalHeaderLabels([
            "Service Date", "Machine", "Action Taken", "Parts Replaced", "Downtime (hrs)", "Technician"
        ])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_history.verticalHeader().setVisible(False)
        hist_layout.addWidget(self.tbl_history)

        self.tabs.addTab(tab_hist, "📋 Maintenance History")

        main_layout.addWidget(self.tabs)

    def load_work_orders(self) -> None:
        """Populate active work orders table."""
        s_filter = self.cb_f_status.currentText()
        p_filter = self.cb_f_priority.currentText()
        orders = MaintenanceService.get_all_work_orders(status_filter=s_filter, priority_filter=p_filter)

        self.tbl_orders.setRowCount(len(orders))
        for row, wo in enumerate(orders):
            self.tbl_orders.setItem(row, 0, QTableWidgetItem(str(wo["id"])))
            self.tbl_orders.setItem(row, 1, QTableWidgetItem(wo["code"]))
            self.tbl_orders.setItem(row, 2, QTableWidgetItem(f"{wo['machine_code']} ({wo['machine_name'][:12]})"))
            self.tbl_orders.setItem(row, 3, QTableWidgetItem(wo["title"]))

            p_item = QTableWidgetItem(wo["priority"])
            p_item.setForeground(Qt.red if wo["priority"] in ["High", "Emergency"] else Qt.yellow if wo["priority"] == "Medium" else Qt.cyan)
            self.tbl_orders.setItem(row, 4, p_item)

            s_item = QTableWidgetItem(wo["status"])
            s_item.setForeground(Qt.green if wo["status"] == "Completed" else Qt.yellow if wo["status"] == "In Progress" else Qt.white)
            self.tbl_orders.setItem(row, 5, s_item)

            self.tbl_orders.setItem(row, 6, QTableWidgetItem(f"{wo['estimated_downtime_hours']} hrs"))
            self.tbl_orders.setItem(row, 7, QTableWidgetItem(wo["scheduled_start"]))

    def load_ai_suggestions(self) -> None:
        """Populate AI adaptive suggestions table."""
        suggestions = MaintenanceService.generate_adaptive_suggestions()
        self.tbl_suggestions.setRowCount(len(suggestions))
        for row, s in enumerate(suggestions):
            self.tbl_suggestions.setItem(row, 0, QTableWidgetItem(f"{s['machine_code']} - {s['machine_name'][:12]}"))
            self.tbl_suggestions.setItem(row, 1, QTableWidgetItem(s["title"]))
            p_item = QTableWidgetItem(s["priority"])
            p_item.setForeground(Qt.red if s["priority"] in ["High", "Emergency"] else Qt.yellow)
            self.tbl_suggestions.setItem(row, 2, p_item)
            self.tbl_suggestions.setItem(row, 3, QTableWidgetItem(f"{s['suggested_downtime_hours']} hrs"))
            self.tbl_suggestions.setItem(row, 4, QTableWidgetItem(s["trigger"]))

            btn_convert = QPushButton("Create Work Order")
            btn_convert.setFixedHeight(24)
            btn_convert.clicked.connect(lambda checked=False, data=s: self._convert_suggestion(data))
            self.tbl_suggestions.setCellWidget(row, 5, btn_convert)

    def load_drift_events(self) -> None:
        """Populate drift events table."""
        events = MaintenanceService.get_drift_events()
        self.tbl_drift.setRowCount(len(events))
        for row, e in enumerate(events):
            self.tbl_drift.setItem(row, 0, QTableWidgetItem(str(e["id"])))
            self.tbl_drift.setItem(row, 1, QTableWidgetItem(f"{e['model_name']} ({e['sensor_code']})"))
            self.tbl_drift.setItem(row, 2, QTableWidgetItem(e["detector"]))
            self.tbl_drift.setItem(row, 3, QTableWidgetItem(f"{e['statistic']:.4f}"))
            self.tbl_drift.setItem(row, 4, QTableWidgetItem(f"{e['threshold']:.4f}"))
            st_item = QTableWidgetItem(e["status"])
            st_item.setForeground(Qt.yellow if e["status"] == "detected" else Qt.green)
            self.tbl_drift.setItem(row, 5, st_item)
            self.tbl_drift.setItem(row, 6, QTableWidgetItem(e["recommendation"]))

    def load_history(self) -> None:
        """Populate maintenance history table."""
        records = MaintenanceService.get_maintenance_history()
        self.tbl_history.setRowCount(len(records))
        for row, r in enumerate(records):
            self.tbl_history.setItem(row, 0, QTableWidgetItem(r["service_date"]))
            self.tbl_history.setItem(row, 1, QTableWidgetItem(f"{r['machine_code']} ({r['machine_name'][:12]})"))
            self.tbl_history.setItem(row, 2, QTableWidgetItem(r["action_taken"]))
            self.tbl_history.setItem(row, 3, QTableWidgetItem(r["parts_replaced"]))
            self.tbl_history.setItem(row, 4, QTableWidgetItem(f"{r['downtime_hours']} hrs"))
            self.tbl_history.setItem(row, 5, QTableWidgetItem(r["technician"]))

    def _create_new_work_order(self) -> None:
        dlg = WorkOrderDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            MaintenanceService.create_work_order(
                machine_code=data["machine_code"],
                title=data["title"],
                description=data["description"],
                priority=data["priority"],
                estimated_downtime_hours=data["estimated_downtime_hours"],
            )
            self.load_work_orders()

    def _convert_suggestion(self, sugg_data: Dict[str, Any]) -> None:
        dlg = WorkOrderDialog(self, initial_data=sugg_data)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            MaintenanceService.create_work_order(
                machine_code=data["machine_code"],
                title=data["title"],
                description=data["description"],
                priority=data["priority"],
                estimated_downtime_hours=data["estimated_downtime_hours"],
            )
            self.load_work_orders()
            self.tabs.setCurrentIndex(0)

    def _approve_selected(self) -> None:
        row = self.tbl_orders.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select Row", "Please select a work order to approve.")
            return
        wo_id = int(self.tbl_orders.item(row, 0).text())
        if MaintenanceService.approve_work_order(wo_id):
            self.load_work_orders()

    def _complete_selected(self) -> None:
        row = self.tbl_orders.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select Row", "Please select a work order to complete.")
            return
        wo_id = int(self.tbl_orders.item(row, 0).text())
        wo_title = self.tbl_orders.item(row, 3).text()

        # Mark completed with history entry
        MaintenanceService.complete_work_order(
            work_order_id=wo_id,
            actual_downtime_hours=2.0,
            action_taken=f"Executed task: {wo_title}",
            parts_replaced="Inspection verified; calibrated",
            technician_name="Maintenance Engineer",
        )
        self.load_work_orders()
        self.load_history()
