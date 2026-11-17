"""
IndusPredict AI - Live Sensor Monitoring & Ingestion View.
Provides real-time multi-channel sensor waveform charts, active telemetry cards,
threshold configuration, abnormal reading alerts, CSV/Excel dataset upload,
Paho MQTT connectivity, and multi-mode physics simulation.
"""

from collections import deque
from datetime import datetime
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.config import SENSOR_SPECS
from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.iot.csv_uploader import CSVTelemetryUploader
from induspredict.iot.mqtt_client import MQTTTelemetryClient
from induspredict.iot.opcua_client import OPCUATelemetryClient
from induspredict.iot.simulator import SensorSimulator, SimulatorWorker
from induspredict.services.machine_service import MachineService
from induspredict.services.telemetry_service import TelemetryService
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

logger = get_logger("ui.live_monitoring")


class ThresholdConfigDialog(QDialog):
    """Dialog allowing maintenance engineers to adjust sensor thresholds."""

    def __init__(self, machine_code: str, sensors: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Threshold Limits Configuration – Asset {machine_code}")
        self.setModal(True)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        title_lbl = QLabel(f"Configure Operational Limits for {machine_code}")
        title_lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {CYAN_PRIMARY};")
        layout.addWidget(title_lbl)

        self.sensor_combos = []
        form = QFormLayout()
        form.setSpacing(10)

        self.cb_sensor = QComboBox()
        for s in sensors:
            self.cb_sensor.addItem(f"{s['name']} ({s['code']})", s)
        self.cb_sensor.currentIndexChanged.connect(self._on_sensor_changed)
        form.addRow("Target Sensor:", self.cb_sensor)

        self.spin_warn = QDoubleSpinBox()
        self.spin_warn.setRange(-1000.0, 50000.0)
        self.spin_warn.setDecimals(2)
        form.addRow("Warning Threshold:", self.spin_warn)

        self.spin_crit = QDoubleSpinBox()
        self.spin_crit.setRange(-1000.0, 50000.0)
        self.spin_crit.setDecimals(2)
        form.addRow("Critical Threshold:", self.spin_crit)

        layout.addLayout(form)
        self._on_sensor_changed()

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "secondary-btn")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_save = QPushButton("Save Thresholds to DB")
        self.btn_save.setProperty("class", "primary-btn")
        self.btn_save.clicked.connect(self._handle_save)

        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_save)
        layout.addLayout(btn_box)

    def _on_sensor_changed(self) -> None:
        sensor_data = self.cb_sensor.currentData()
        if sensor_data:
            self.spin_warn.setValue(float(sensor_data.get("warning_threshold", 80.0)))
            self.spin_crit.setValue(float(sensor_data.get("critical_threshold", 100.0)))
            unit = sensor_data.get("unit", "")
            self.spin_warn.setSuffix(f" {unit}")
            self.spin_crit.setSuffix(f" {unit}")

    def _handle_save(self) -> None:
        sensor_data = self.cb_sensor.currentData()
        if not sensor_data:
            return

        w_val = self.spin_warn.value()
        c_val = self.spin_crit.value()

        if w_val >= c_val:
            QMessageBox.warning(self, "Validation Error", "Warning threshold must be strictly lower than Critical threshold.")
            return

        res = TelemetryService.update_sensor_thresholds(
            sensor_id=sensor_data["id"],
            warning_val=w_val,
            critical_val=c_val,
        )

        if res["success"]:
            QMessageBox.information(self, "Success", f"Thresholds for {sensor_data['code']} successfully updated.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", res.get("error", "Update failed."))


class LiveSensorCard(QFrame):
    """Card displaying a single sensor parameter with live value, threshold, and status badge."""

    def __init__(self, sensor_type: str, title: str, unit: str, parent=None):
        super().__init__(parent)
        self.sensor_type = sensor_type
        self.unit = unit
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # Header
        top_row = QHBoxLayout()
        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_badge = QLabel("NOMINAL")
        self.lbl_badge.setStyleSheet(f"background-color: rgba(46, 204, 113, 0.15); color: {COLOR_HEALTHY}; border: 1px solid {COLOR_HEALTHY}; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;")
        top_row.addWidget(self.lbl_title)
        top_row.addStretch()
        top_row.addWidget(self.lbl_badge)
        layout.addLayout(top_row)

        # Value
        self.lbl_val = QLabel(f"-- {unit}")
        self.lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {CYAN_PRIMARY};")
        layout.addWidget(self.lbl_val)

        # Progress / Threshold bar
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(35)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setFixedHeight(5)
        self.prog_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_PANEL};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {CYAN_PRIMARY};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.prog_bar)

        # Threshold caption
        self.lbl_thresh = QLabel("Warn: -- | Crit: --")
        self.lbl_thresh.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED};")
        layout.addWidget(self.lbl_thresh)

    def update_reading(self, value: float, status: str, warning_th: float, critical_th: float) -> None:
        self.lbl_val.setText(f"{value:.2f} {self.unit}")
        self.lbl_thresh.setText(f"Warn: {warning_th:.1f} | Crit: {critical_th:.1f} {self.unit}")

        # Update badge & bar colors based on status
        pct = min(100, int((value / max(1.0, critical_th)) * 100))
        self.prog_bar.setValue(pct)

        if status == "critical":
            self.lbl_badge.setText("CRITICAL")
            self.lbl_badge.setStyleSheet(f"background-color: rgba(231, 76, 60, 0.15); color: {COLOR_CRITICAL}; border: 1px solid {COLOR_CRITICAL}; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;")
            self.lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLOR_CRITICAL};")
        elif status == "warning":
            self.lbl_badge.setText("WARNING")
            self.lbl_badge.setStyleSheet(f"background-color: rgba(243, 156, 18, 0.15); color: {COLOR_WARNING}; border: 1px solid {COLOR_WARNING}; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;")
            self.lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLOR_WARNING};")
        elif status == "disconnected":
            self.lbl_badge.setText("DISCONNECTED")
            self.lbl_badge.setStyleSheet(f"background-color: rgba(142, 142, 147, 0.15); color: {TEXT_MUTED}; border: 1px solid {TEXT_MUTED}; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;")
            self.lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {TEXT_MUTED};")
        else:
            self.lbl_badge.setText("NOMINAL")
            self.lbl_badge.setStyleSheet(f"background-color: rgba(46, 204, 113, 0.15); color: {COLOR_HEALTHY}; border: 1px solid {COLOR_HEALTHY}; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;")
            self.lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {CYAN_PRIMARY};")


class LiveMonitoringView(QWidget):
    """Industrial real-time sensor telemetry monitoring dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_machine_code = "CMP-01"
        self.telemetry_service = TelemetryService()
        self.csv_uploader = CSVTelemetryUploader()
        self.mqtt_client = MQTTTelemetryClient()
        self.opcua_client = OPCUATelemetryClient()

        # Simulator setup
        self.simulator = SensorSimulator(machine_code=self.current_machine_code, sample_rate_hz=1.0)
        self.sim_worker = SimulatorWorker(self.simulator)

        # Sliding waveforms buffer for charts (last 60 seconds)
        self.waveform_history = {
            "vibration": deque(maxlen=60),
            "temperature": deque(maxlen=60),
            "pressure": deque(maxlen=60),
            "motor_speed": deque(maxlen=60),
        }

        self._init_ui()
        self._connect_signals()
        self.load_machines_list()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Top Control Bar (Machine Selector + Ingestion Source Badge)
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        # Asset Selector
        lbl_mach = QLabel("Monitored Asset:")
        lbl_mach.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_machines = QComboBox()
        self.cb_machines.setMinimumWidth(240)
        self.cb_machines.currentIndexChanged.connect(self._on_machine_changed)
        top_bar.addWidget(lbl_mach)
        top_bar.addWidget(self.cb_machines)

        # Ingestion Source Indicator Badge (Explicit disclosure)
        self.lbl_source_badge = QLabel("SOURCE: PHYSICS SIMULATOR")
        self.lbl_source_badge.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.12); color: {CYAN_PRIMARY}; border: 1px solid {CYAN_PRIMARY}; border-radius: 4px; padding: 4px 12px; font-size: 11px; font-weight: bold;"
        )
        top_bar.addWidget(self.lbl_source_badge)

        top_bar.addStretch()

        # Thresholds Dialog Button
        self.btn_thresholds = QPushButton("⚙️ Threshold Limits")
        self.btn_thresholds.setProperty("class", "secondary-btn")
        self.btn_thresholds.clicked.connect(self._open_threshold_dialog)
        top_bar.addWidget(self.btn_thresholds)

        # CSV Upload Button
        self.btn_upload = QPushButton("📂 Upload CSV / Excel")
        self.btn_upload.setProperty("class", "secondary-btn")
        self.btn_upload.clicked.connect(self._handle_upload_file)
        top_bar.addWidget(self.btn_upload)

        main_layout.addLayout(top_bar)

        # 2. Simulator & Ingestion Controls Toolbar
        sim_toolbar = QFrame()
        sim_toolbar.setProperty("class", "card")
        sim_layout = QHBoxLayout(sim_toolbar)
        sim_layout.setContentsMargins(12, 8, 12, 8)
        sim_layout.setSpacing(10)

        lbl_sim = QLabel("Telemetry Stream Controls:")
        lbl_sim.setStyleSheet(f"font-weight: 600; color: {TEXT_PRIMARY};")
        sim_layout.addWidget(lbl_sim)

        self.btn_sim_toggle = QPushButton("▶ Start Live Telemetry")
        self.btn_sim_toggle.setProperty("class", "primary-btn")
        self.btn_sim_toggle.clicked.connect(self._toggle_simulation)
        sim_layout.addWidget(self.btn_sim_toggle)

        lbl_mode = QLabel("Fault Injection Mode:")
        lbl_mode.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.cb_fault_mode = QComboBox()
        self.cb_fault_mode.addItems(self.simulator.ALL_MODES)
        self.cb_fault_mode.currentIndexChanged.connect(self._on_fault_mode_changed)
        sim_layout.addWidget(lbl_mode)
        sim_layout.addWidget(self.cb_fault_mode)

        self.btn_reset_fault = QPushButton("↺ Reset Nominal")
        self.btn_reset_fault.setProperty("class", "secondary-btn")
        self.btn_reset_fault.clicked.connect(self._reset_nominal)
        sim_layout.addWidget(self.btn_reset_fault)

        sim_layout.addStretch()

        # Industrial Protocols Quick Connect
        self.btn_mqtt = QPushButton("📡 Connect MQTT Broker")
        self.btn_mqtt.setProperty("class", "secondary-btn")
        self.btn_mqtt.clicked.connect(self._handle_mqtt_connect)
        sim_layout.addWidget(self.btn_mqtt)

        main_layout.addWidget(sim_toolbar)

        # 3. Sensor Cards Grid (7 variables)
        cards_grid = QGridLayout()
        cards_grid.setSpacing(12)

        self.card_temp = LiveSensorCard("temperature", "Temperature", "°C")
        self.card_vib = LiveSensorCard("vibration", "Vibration Velocity", "mm/s")
        self.card_press = LiveSensorCard("pressure", "Operating Pressure", "bar")
        self.card_speed = LiveSensorCard("motor_speed", "Motor Speed", "RPM")
        self.card_curr = LiveSensorCard("current", "Motor Current", "A")
        self.card_torq = LiveSensorCard("torque", "Shaft Torque", "Nm")
        self.card_hours = LiveSensorCard("operating_hours", "Cumulative Hours", "hrs")

        cards_grid.addWidget(self.card_temp, 0, 0)
        cards_grid.addWidget(self.card_vib, 0, 1)
        cards_grid.addWidget(self.card_press, 0, 2)
        cards_grid.addWidget(self.card_speed, 0, 3)
        cards_grid.addWidget(self.card_curr, 1, 0)
        cards_grid.addWidget(self.card_torq, 1, 1)
        cards_grid.addWidget(self.card_hours, 1, 2)

        main_layout.addLayout(cards_grid)

        # 4. Waveform Charts & Live Alerts Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Waveform Chart
        self.chart_live = ChartCard("Real-Time Multi-Sensor Waveforms (Rolling 60s Buffer)")
        splitter.addWidget(self.chart_live)

        # Alerts Feed & Telemetry Table Container
        right_panel = QFrame()
        right_panel.setProperty("class", "card")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        lbl_alert_header = QLabel("LIVE THRESHOLD VIOLATIONS & ALARMS")
        lbl_alert_header.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {COLOR_CRITICAL};")
        right_layout.addWidget(lbl_alert_header)

        self.tbl_live_alerts = QTableWidget()
        self.tbl_live_alerts.setColumnCount(4)
        self.tbl_live_alerts.setHorizontalHeaderLabels(["Time", "Sensor", "Value", "Severity"])
        self.tbl_live_alerts.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_live_alerts.verticalHeader().setVisible(False)
        self.tbl_live_alerts.setMinimumWidth(340)
        right_layout.addWidget(self.tbl_live_alerts)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)

        # 5. Timer for Redrawing Live Waveform Chart at 2 Hz
        self.chart_timer = QTimer(self)
        self.chart_timer.setInterval(500)
        self.chart_timer.timeout.connect(self._redraw_waveform_chart)
        self.chart_timer.start()

    def _connect_signals(self) -> None:
        """Connect IoT workers and services to UI slots."""
        # Simulator
        self.simulator.frame_generated.connect(self._on_telemetry_frame)
        self.simulator.alert_triggered.connect(self._on_live_alert)

        # MQTT
        self.mqtt_client.telemetry_received.connect(self._on_telemetry_frame)
        self.mqtt_client.connected_signal.connect(lambda h, p: self._update_source_badge(f"SOURCE: MQTT ({h}:{p})", COLOR_HEALTHY))
        self.mqtt_client.disconnected_signal.connect(lambda r: self._update_source_badge("SOURCE: MQTT DISCONNECTED", COLOR_WARNING))

        # CSV Uploader Playback
        self.csv_uploader.playback_frame.connect(self._on_telemetry_frame)

    def load_machines_list(self) -> None:
        """Populate machine dropdown with registered assets."""
        self.cb_machines.blockSignals(True)
        self.cb_machines.clear()
        machines = MachineService.get_all_machines()
        for m in machines:
            self.cb_machines.addItem(f"{m['code']} - {m['name'][:24]}...", m["code"])
        self.cb_machines.blockSignals(False)

        if self.cb_machines.count() > 0:
            self.current_machine_code = self.cb_machines.currentData()
            self.simulator.set_machine(self.current_machine_code)

    def _on_machine_changed(self, idx: int) -> None:
        self.current_machine_code = self.cb_machines.currentData()
        self.simulator.set_machine(self.current_machine_code)
        # Clear local chart buffers
        for b in self.waveform_history.values():
            b.clear()

    def _update_source_badge(self, text: str, color: str) -> None:
        self.lbl_source_badge.setText(text)
        self.lbl_source_badge.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.12); color: {color}; border: 1px solid {color}; border-radius: 4px; padding: 4px 12px; font-size: 11px; font-weight: bold;"
        )

    def _toggle_simulation(self) -> None:
        if self.sim_worker.isRunning():
            self.sim_worker.stop()
            self.btn_sim_toggle.setText("▶ Start Live Telemetry")
            self.btn_sim_toggle.setProperty("class", "primary-btn")
            self._update_source_badge("SOURCE: PAUSED / OFFLINE", TEXT_MUTED)
        else:
            self.sim_worker.start()
            self.btn_sim_toggle.setText("⏸ Pause Live Telemetry")
            self.btn_sim_toggle.setProperty("class", "secondary-btn")
            self._update_source_badge("SOURCE: PHYSICS SIMULATOR", CYAN_PRIMARY)

    def _on_fault_mode_changed(self, idx: int) -> None:
        mode = self.cb_fault_mode.currentText()
        self.simulator.set_mode(mode)

    def _reset_nominal(self) -> None:
        self.simulator.reset_to_nominal()
        self.cb_fault_mode.setCurrentText(SensorSimulator.MODE_NOMINAL)

    def _on_telemetry_frame(self, frame: Dict[str, Any]) -> None:
        """Process real-time frame from Simulator, MQTT, or CSV."""
        if frame.get("machine_code") != self.current_machine_code:
            return

        sensors = frame.get("sensors", {})
        ts_str = frame.get("timestamp", utc_now().strftime("%Y-%m-%d %H:%M:%S"))

        # Ingest into TelemetryService for SQLite persistence and threshold checking
        self.telemetry_service.ingest_frame(frame)

        # Update Live Sensor Cards
        mapping = {
            "temperature": self.card_temp,
            "vibration": self.card_vib,
            "pressure": self.card_press,
            "motor_speed": self.card_speed,
            "current": self.card_curr,
            "torque": self.card_torq,
            "operating_hours": self.card_hours,
        }

        for stype, card in mapping.items():
            sdata = sensors.get(stype)
            if sdata:
                val = float(sdata.get("value", 0.0))
                stat = sdata.get("status", "healthy")
                spec = SENSOR_SPECS.get(stype, {"warning": 80, "critical": 100})
                card.update_reading(val, stat, spec["warning"], spec["critical"])

                if stype in self.waveform_history:
                    self.waveform_history[stype].append(val)

    def _on_live_alert(self, alert_data: Dict[str, Any]) -> None:
        """Add threshold violation alert to the live table feed."""
        row_pos = 0
        self.tbl_live_alerts.insertRow(row_pos)
        self.tbl_live_alerts.setItem(row_pos, 0, QTableWidgetItem(alert_data.get("timestamp", "")[-8:]))
        self.tbl_live_alerts.setItem(row_pos, 1, QTableWidgetItem(str(alert_data.get("sensor_type", ""))))
        self.tbl_live_alerts.setItem(row_pos, 2, QTableWidgetItem(f"{alert_data.get('value', 0):.2f} {alert_data.get('unit', '')}"))

        sev_item = QTableWidgetItem(str(alert_data.get("severity", "")))
        if alert_data.get("severity") == "Critical":
            sev_item.setForeground(Qt.red)
        else:
            sev_item.setForeground(Qt.yellow)
        self.tbl_live_alerts.setItem(row_pos, 3, sev_item)

        # Cap table to 50 rows
        if self.tbl_live_alerts.rowCount() > 50:
            self.tbl_live_alerts.removeRow(self.tbl_live_alerts.rowCount() - 1)

    def _redraw_waveform_chart(self) -> None:
        """Refresh live waveform chart canvas using in-memory deque buffers."""
        ax = self.chart_live.ax
        ax.clear()
        self.chart_live._style_axis(ax)

        has_data = any(len(b) > 0 for b in self.waveform_history.values())
        if not has_data:
            ax.text(
                0.5, 0.5, "Live Waveform Stream Awaiting Telemetry...",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, color=TEXT_MUTED, fontsize=11
            )
            self.chart_live.canvas.draw()
            return

        # Plot Normalized Waveforms
        colors = {
            "vibration": CYAN_PRIMARY,
            "temperature": COLOR_WARNING,
            "pressure": TEAL_ACCENT,
            "motor_speed": COLOR_INFO,
        }

        for stype, buf in self.waveform_history.items():
            if len(buf) > 1:
                vals = list(buf)
                spec = SENSOR_SPECS.get(stype, {"max": 100})
                norm_vals = [v / max(1.0, spec["max"]) for v in vals]
                x_axis = np.arange(len(norm_vals))
                ax.plot(x_axis, norm_vals, color=colors.get(stype, CYAN_PRIMARY), linewidth=1.8, label=f"{stype.title()} (Norm)")

        ax.set_ylabel("Normalized Envelope [0.0 - 1.0]", color=TEXT_SECONDARY, fontsize=9)
        ax.set_xlabel("Recent Samples (Sliding Window)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_SUBTLE, labelcolor=TEXT_PRIMARY, fontsize=8, loc="upper right")

        self.chart_live.figure.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.20)
        self.chart_live.canvas.draw()

    def _open_threshold_dialog(self) -> None:
        """Launch modal for configuring sensor limits."""
        details = MachineService.get_machine_details_by_code(self.current_machine_code)
        if not details or not details.get("sensors"):
            # Fallback by querying all machines
            machines = MachineService.get_all_machines()
            target_m = next((m for m in machines if m["code"] == self.current_machine_code), None)
            if target_m:
                details = MachineService.get_machine_details(target_m["id"])

        if details and details.get("sensors"):
            dlg = ThresholdConfigDialog(self.current_machine_code, details["sensors"], parent=self)
            dlg.exec()
        else:
            QMessageBox.warning(self, "No Sensors", f"No configured sensors found for {self.current_machine_code}.")

    def _handle_upload_file(self) -> None:
        """File picker for uploading industrial CSV or Excel datasets."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Industrial Telemetry Dataset (CSV / Excel)",
            "",
            "Industrial Datasets (*.csv *.xlsx *.xls)",
        )
        if not file_path:
            return

        success, df, msg = self.csv_uploader.parse_and_validate_file(file_path)
        if not success:
            QMessageBox.critical(self, "Dataset Error", msg)
            return

        reply = QMessageBox.question(
            self,
            "Confirm Dataset Ingestion",
            f"Successfully validated dataset:\n\n{msg}\n\nIngest {len(df)} records into SQLite for asset '{self.current_machine_code}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            summary = self.csv_uploader.import_dataframe_to_db(df, self.current_machine_code)
            if summary["success"]:
                self._update_source_badge("SOURCE: UPLOADED DATASET", "#A371F7")
                QMessageBox.information(
                    self,
                    "Ingestion Complete",
                    f"Successfully imported {summary['readings_imported']} telemetry points for {self.current_machine_code}.\nReadings tagged as source='uploaded'.",
                )
            else:
                QMessageBox.critical(self, "Import Error", summary.get("error", "Import failed."))

    def _handle_mqtt_connect(self) -> None:
        """Connect to local or remote MQTT broker."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Connect Industrial MQTT Broker")
        dlg.setMinimumWidth(380)

        d_layout = QVBoxLayout(dlg)
        f_layout = QFormLayout()

        txt_host = QLineEdit("test.mosquitto.org")
        txt_port = QLineEdit("1883")
        txt_topic = QLineEdit(f"induspredict/{self.current_machine_code}/telemetry")

        f_layout.addRow("Broker Host:", txt_host)
        f_layout.addRow("Broker Port:", txt_port)
        f_layout.addRow("Subscription Topic:", txt_topic)
        d_layout.addLayout(f_layout)

        btn_box = QHBoxLayout()
        btn_connect = QPushButton("Connect (Read-Only)")
        btn_connect.setProperty("class", "primary-btn")
        btn_connect.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_connect)
        d_layout.addLayout(btn_box)

        if dlg.exec():
            host = txt_host.text().strip()
            port = int(txt_port.text().strip())
            topic = txt_topic.text().strip()
            self.mqtt_client.connect_broker(host=host, port=port, topic=topic)
            self._update_source_badge(f"SOURCE: MQTT ({host})", COLOR_HEALTHY)
