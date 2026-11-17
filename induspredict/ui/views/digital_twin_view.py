"""
IndusPredict AI - Digital Twin Simulator View (Page 7).
Provides an interactive virtual replica of registered machinery, allowing engineers to
perturb operating conditions (Temperature, Vibration, Speed, Pressure, Dynamic Load),
observe coupled ML risk shifts and health degradation, and save what-if scenarios.
"""

from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.logger import get_logger
from induspredict.services.digital_twin_service import DigitalTwinService
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

logger = get_logger("ui.digital_twin")


class DigitalTwinView(QWidget):
    """Page 7: Digital Twin Simulator View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_machine_code = "CMP-01"
        self.last_sim_result: Optional[Dict[str, Any]] = None
        self._init_ui()
        self.load_machine_profile()
        self.load_saved_scenarios()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Digital Twin & What-If Simulation Environment")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Parametric stress perturbation, physics-informed degradation modeling, and ML risk forecasting")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()

        # Machine Selector Dropdown
        m_box = QHBoxLayout()
        m_box.setSpacing(8)
        lbl_m = QLabel("Active Asset Replica:")
        lbl_m.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_machines = QComboBox()
        self.cb_machines.setMinimumWidth(220)
        machines = MachineService.get_all_machines()
        for m in machines:
            self.cb_machines.addItem(f"{m['code']} - {m['name']}", m["code"])
        self.cb_machines.currentIndexChanged.connect(self._on_machine_changed)
        m_box.addWidget(lbl_m)
        m_box.addWidget(self.cb_machines)
        header_row.addLayout(m_box)

        main_layout.addLayout(header_row)

        # Disclaimer Banner (Scientific Integrity)
        banner_frame = QFrame()
        banner_frame.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 6px; padding: 4px;"
        )
        b_lay = QHBoxLayout(banner_frame)
        b_lay.setContentsMargins(10, 6, 10, 6)
        lbl_disc = QLabel(f"⚠️ {DigitalTwinService.DISCLAIMER}")
        lbl_disc.setWordWrap(True)
        lbl_disc.setStyleSheet(f"color: {CYAN_PRIMARY}; font-size: 11px;")
        b_lay.addWidget(lbl_disc)
        main_layout.addWidget(banner_frame)

        # Main Tabs
        self.tabs = QTabWidget()

        # TAB 1: What-If Simulation Studio
        tab_sim = QWidget()
        sim_layout = QHBoxLayout(tab_sim)
        sim_layout.setContentsMargins(12, 12, 12, 12)
        sim_layout.setSpacing(14)

        # Left Column: Operating Parameter Adjustments
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_panel.setProperty("class", "card")
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(14, 14, 14, 14)
        left_lay.setSpacing(12)

        lbl_controls_hdr = QLabel("Perturbation Parameters (What-If)")
        lbl_controls_hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        left_lay.addWidget(lbl_controls_hdr)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft)

        # Temperature
        self.spin_air_t = QDoubleSpinBox()
        self.spin_air_t.setRange(-20.0, 70.0)
        self.spin_air_t.setValue(25.0)
        self.spin_air_t.setSuffix(" °C")
        form.addRow("Ambient Air Temp:", self.spin_air_t)

        self.spin_proc_t = QDoubleSpinBox()
        self.spin_proc_t.setRange(0.0, 150.0)
        self.spin_proc_t.setValue(35.0)
        self.spin_proc_t.setSuffix(" °C")
        form.addRow("Process / Core Temp:", self.spin_proc_t)

        # Rotational Speed
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(500.0, 3500.0)
        self.spin_speed.setValue(1500.0)
        self.spin_speed.setSingleStep(50.0)
        self.spin_speed.setSuffix(" RPM")
        form.addRow("Rotational Speed:", self.spin_speed)

        # Torque / Mechanical Load
        self.spin_torque = QDoubleSpinBox()
        self.spin_torque.setRange(5.0, 100.0)
        self.spin_torque.setValue(40.0)
        self.spin_torque.setSingleStep(2.0)
        self.spin_torque.setSuffix(" Nm")
        form.addRow("Dynamic Torque / Load:", self.spin_torque)

        # Tool / Component Wear
        self.spin_wear = QDoubleSpinBox()
        self.spin_wear.setRange(0.0, 300.0)
        self.spin_wear.setValue(20.0)
        self.spin_wear.setSingleStep(10.0)
        self.spin_wear.setSuffix(" min")
        form.addRow("Component Wear:", self.spin_wear)

        # Vibration
        self.spin_vib = QDoubleSpinBox()
        self.spin_vib.setRange(0.1, 15.0)
        self.spin_vib.setValue(1.45)
        self.spin_vib.setSingleStep(0.2)
        self.spin_vib.setSuffix(" mm/s")
        form.addRow("Vibration Velocity:", self.spin_vib)

        # Pressure
        self.spin_press = QDoubleSpinBox()
        self.spin_press.setRange(0.5, 25.0)
        self.spin_press.setValue(6.8)
        self.spin_press.setSingleStep(0.5)
        self.spin_press.setSuffix(" bar")
        form.addRow("System Pressure:", self.spin_press)

        left_lay.addLayout(form)

        # Quick Preset Stress Buttons
        lbl_presets = QLabel("Simulate Stress Profiles:")
        lbl_presets.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY}; margin-top: 6px;")
        left_lay.addWidget(lbl_presets)

        preset_grid = QGridLayout()
        preset_grid.setSpacing(6)

        btn_p_nominal = QPushButton("Standard Baseline")
        btn_p_nominal.clicked.connect(self._preset_baseline)
        preset_grid.addWidget(btn_p_nominal, 0, 0)

        btn_p_overheat = QPushButton("Thermal Overload")
        btn_p_overheat.clicked.connect(self._preset_thermal)
        preset_grid.addWidget(btn_p_overheat, 0, 1)

        btn_p_high_load = QPushButton("Heavy Overstrain")
        btn_p_high_load.clicked.connect(self._preset_heavy_load)
        preset_grid.addWidget(btn_p_high_load, 1, 0)

        btn_p_severe = QPushButton("Degraded Severe")
        btn_p_severe.clicked.connect(self._preset_severe)
        preset_grid.addWidget(btn_p_severe, 1, 1)

        left_lay.addLayout(preset_grid)

        # Action Buttons
        self.btn_run_sim = QPushButton("⚡ Execute What-If Simulation")
        self.btn_run_sim.setProperty("class", "btn-primary")
        self.btn_run_sim.setFixedHeight(38)
        self.btn_run_sim.clicked.connect(self.run_simulation)
        left_lay.addWidget(self.btn_run_sim)

        self.btn_save_scenario = QPushButton("💾 Save Simulation Scenario")
        self.btn_save_scenario.setFixedHeight(34)
        self.btn_save_scenario.clicked.connect(self._show_save_dialog)
        left_lay.addWidget(self.btn_save_scenario)

        left_lay.addStretch()
        sim_layout.addWidget(left_panel)

        # Right Area: Comparative Results & Charts
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Comparison Cards Row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        # Card 1: Machine Health Shift
        self.card_health = QFrame()
        self.card_health.setProperty("class", "card")
        h_lay = QVBoxLayout(self.card_health)
        h_lay.setContentsMargins(14, 12, 14, 12)
        lbl_h_title = QLabel("VIRTUAL HEALTH COMPARISON")
        lbl_h_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_base_health = QLabel("Baseline: 92.5%")
        self.lbl_base_health.setStyleSheet(f"color: {COLOR_HEALTHY}; font-size: 13px; font-weight: bold;")
        self.lbl_sim_health = QLabel("Simulated: 78.4%")
        self.lbl_sim_health.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 18px; font-weight: bold;")
        self.lbl_health_delta = QLabel("Delta: -14.1% Degradation")
        self.lbl_health_delta.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        h_lay.addWidget(lbl_h_title)
        h_lay.addWidget(self.lbl_base_health)
        h_lay.addWidget(self.lbl_sim_health)
        h_lay.addWidget(self.lbl_health_delta)
        cards_row.addWidget(self.card_health, 1)

        # Card 2: Failure Probability Shift
        self.card_risk = QFrame()
        self.card_risk.setProperty("class", "card")
        r_lay = QVBoxLayout(self.card_risk)
        r_lay.setContentsMargins(14, 12, 14, 12)
        lbl_r_title = QLabel("PREDICTED RISK COMPARISON")
        lbl_r_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_base_risk = QLabel("Baseline: 1.2% (Normal)")
        self.lbl_base_risk.setStyleSheet(f"color: {COLOR_HEALTHY}; font-size: 13px; font-weight: bold;")
        self.lbl_sim_risk = QLabel("Simulated: 42.8% (Warning)")
        self.lbl_sim_risk.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 18px; font-weight: bold;")
        self.lbl_risk_delta = QLabel("Risk Delta: +41.6%")
        self.lbl_risk_delta.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        r_lay.addWidget(lbl_r_title)
        r_lay.addWidget(self.lbl_base_risk)
        r_lay.addWidget(self.lbl_sim_risk)
        r_lay.addWidget(self.lbl_risk_delta)
        cards_row.addWidget(self.card_risk, 1)

        # Card 3: Physical Diagnostic Modes
        self.card_diag = QFrame()
        self.card_diag.setProperty("class", "card")
        d_lay = QVBoxLayout(self.card_diag)
        d_lay.setContentsMargins(14, 12, 14, 12)
        lbl_d_title = QLabel("PHYSICAL STRESS DIAGNOSTIC")
        lbl_d_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_diag_mode = QLabel("Nominal Operational Envelope")
        self.lbl_diag_mode.setStyleSheet(f"color: {COLOR_HEALTHY}; font-size: 14px; font-weight: bold;")
        self.lbl_diag_power = QLabel("Mech Power: 6.28 kW | ΔT: 10.0 °C")
        self.lbl_diag_power.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.lbl_diag_overstrain = QLabel("Overstrain: 800 Nm·min")
        self.lbl_diag_overstrain.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        d_lay.addWidget(lbl_d_title)
        d_lay.addWidget(self.lbl_diag_mode)
        d_lay.addWidget(self.lbl_diag_power)
        d_lay.addWidget(self.lbl_diag_overstrain)
        cards_row.addWidget(self.card_diag, 1)

        right_panel.addLayout(cards_row)

        # Charts: Comparison Bar Chart
        self.chart_comparison = ChartCard("Baseline vs. Simulated Operating Stress Envelope")
        right_panel.addWidget(self.chart_comparison, 1)

        sim_layout.addLayout(right_panel, 1)
        self.tabs.addTab(tab_sim, "⚡ What-If Simulation Studio")

        # TAB 2: Saved Scenarios & History
        tab_scenarios = QWidget()
        scen_layout = QVBoxLayout(tab_scenarios)
        scen_layout.setContentsMargins(16, 16, 16, 16)
        scen_layout.setSpacing(12)

        lbl_s_hdr = QLabel("Archived Digital Twin What-If Scenarios")
        lbl_s_hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        scen_layout.addWidget(lbl_s_hdr)

        self.tbl_scenarios = QTableWidget()
        self.tbl_scenarios.setColumnCount(7)
        self.tbl_scenarios.setHorizontalHeaderLabels([
            "ID", "Machine", "Scenario Name", "Risk Delta", "Health Delta", "Saved At", "Notes"
        ])
        self.tbl_scenarios.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_scenarios.verticalHeader().setVisible(False)
        scen_layout.addWidget(self.tbl_scenarios)

        self.tabs.addTab(tab_scenarios, "📁 Archived Scenarios")

        main_layout.addWidget(self.tabs)

    def _on_machine_changed(self, idx: int) -> None:
        code = self.cb_machines.currentData()
        if code:
            self.current_machine_code = code
            self.load_machine_profile()
            self.load_saved_scenarios()

    def load_machine_profile(self) -> None:
        """Load virtual profile and baseline telemetry for selected machine."""
        try:
            profile = DigitalTwinService.get_machine_virtual_profile(self.current_machine_code)
            base = profile["baseline_telemetry"]

            self.spin_air_t.setValue(base["air_temperature"])
            self.spin_proc_t.setValue(base["process_temperature"])
            self.spin_speed.setValue(base["rotational_speed"])
            self.spin_torque.setValue(base["torque"])
            self.spin_wear.setValue(base["tool_wear"])
            self.spin_vib.setValue(base["vibration"])
            self.spin_press.setValue(base["pressure"])

            # Run initial simulation with baseline
            self.run_simulation()
        except Exception as e:
            logger.error(f"Error loading virtual profile: {e}")

    def run_simulation(self) -> None:
        """Execute what-if simulation comparing baseline vs current spinbox parameters."""
        try:
            sim_params = {
                "air_temperature": self.spin_air_t.value(),
                "process_temperature": self.spin_proc_t.value(),
                "rotational_speed": self.spin_speed.value(),
                "torque": self.spin_torque.value(),
                "tool_wear": self.spin_wear.value(),
                "vibration": self.spin_vib.value(),
                "pressure": self.spin_press.value(),
            }

            res = DigitalTwinService.simulate_what_if(
                self.current_machine_code,
                sim_params,
                model_name="Random Forest",
            )
            self.last_sim_result = res

            # Update KPI cards
            b_health = res["baseline"]["health_percentage"]
            s_health = res["simulated"]["health_percentage"]
            h_delta = res["comparison"]["estimated_health_delta_pct"]

            self.lbl_base_health.setText(f"Baseline: {b_health}%")
            self.lbl_sim_health.setText(f"Simulated: {s_health}%")
            self.lbl_sim_health.setStyleSheet(
                f"color: {COLOR_HEALTHY if s_health > 75 else COLOR_WARNING if s_health > 50 else COLOR_CRITICAL}; font-size: 18px; font-weight: bold;"
            )
            self.lbl_health_delta.setText(f"Health Delta: {h_delta:+.1f}%")

            b_risk = res["baseline"]["failure_probability"] * 100.0
            s_risk = res["simulated"]["failure_probability"] * 100.0
            r_delta = res["comparison"]["predicted_risk_delta_pct"]

            self.lbl_base_risk.setText(f"Baseline: {b_risk:.1f}% ({res['baseline']['risk_category']})")
            self.lbl_sim_risk.setText(f"Simulated: {s_risk:.1f}% ({res['simulated']['risk_category']})")
            self.lbl_sim_risk.setStyleSheet(
                f"color: {COLOR_HEALTHY if s_risk < 20 else COLOR_WARNING if s_risk < 50 else COLOR_CRITICAL}; font-size: 18px; font-weight: bold;"
            )
            self.lbl_risk_delta.setText(f"Risk Delta: {r_delta:+.1f}%")

            diag_mode = res["simulated"]["likely_failure_mode"]
            self.lbl_diag_mode.setText(diag_mode)
            self.lbl_diag_mode.setStyleSheet(
                f"color: {COLOR_HEALTHY if diag_mode == 'None (Normal Operation)' else COLOR_CRITICAL}; font-size: 13px; font-weight: bold;"
            )
            self.lbl_diag_power.setText(
                f"Mech Power: {res['simulated']['mechanical_power_kw']:.2f} kW | ΔT: {res['simulated']['delta_t_c']:.1f} °C"
            )
            self.lbl_diag_overstrain.setText(f"Overstrain: {res['simulated']['overstrain_index']:.1f} Nm·min")

            # Update Comparative Chart
            self._render_comparison_chart(res)

        except Exception as e:
            logger.error(f"Simulation execution failed: {e}")

    def _render_comparison_chart(self, res: Dict[str, Any]) -> None:
        """Render side-by-side grouped bar chart comparing normalized stress indices."""
        self.chart_comparison.clear()
        ax = self.chart_comparison.get_axes()

        categories = ["Failure Risk %", "Core Temp (°C)", "Speed (RPM/100)", "Torque (Nm)", "Vibration (mm/s)"]
        b_vals = [
            res["baseline"]["failure_probability"] * 100.0,
            res["baseline"]["parameters"]["process_temperature"],
            res["baseline"]["parameters"]["rotational_speed"] / 100.0,
            res["baseline"]["parameters"]["torque"],
            res["baseline"]["parameters"]["vibration"],
        ]
        s_vals = [
            res["simulated"]["failure_probability"] * 100.0,
            res["simulated"]["parameters"]["process_temperature"],
            res["simulated"]["parameters"]["rotational_speed"] / 100.0,
            res["simulated"]["parameters"]["torque"],
            res["simulated"]["parameters"]["vibration"],
        ]

        x = range(len(categories))
        width = 0.35

        ax.bar([i - width / 2 for i in x], b_vals, width, label="Baseline", color="#00C9A7", alpha=0.85)
        ax.bar([i + width / 2 for i in x], s_vals, width, label="Simulated (What-If)", color="#00F2FE", alpha=0.85)

        ax.set_xticks(list(x))
        ax.set_xticklabels(categories, fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("Value / Scaled Magnitude", color=TEXT_SECONDARY, fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_SUBTLE, labelcolor=TEXT_PRIMARY)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        self.chart_comparison.draw()

    def _preset_baseline(self) -> None:
        self.load_machine_profile()

    def _preset_thermal(self) -> None:
        self.spin_air_t.setValue(35.0)
        self.spin_proc_t.setValue(48.0)
        self.spin_speed.setValue(1350.0)
        self.spin_torque.setValue(45.0)
        self.spin_vib.setValue(2.2)
        self.run_simulation()

    def _preset_heavy_load(self) -> None:
        self.spin_speed.setValue(1320.0)
        self.spin_torque.setValue(68.0)
        self.spin_wear.setValue(160.0)
        self.spin_vib.setValue(3.5)
        self.run_simulation()

    def _preset_severe(self) -> None:
        self.spin_air_t.setValue(38.0)
        self.spin_proc_t.setValue(52.0)
        self.spin_speed.setValue(1250.0)
        self.spin_torque.setValue(72.0)
        self.spin_wear.setValue(220.0)
        self.spin_vib.setValue(4.8)
        self.run_simulation()

    def _show_save_dialog(self) -> None:
        """Show prompt to save current simulation scenario."""
        if not self.last_sim_result:
            return

        scenario_name = f"WhatIf_{self.current_machine_code}_{self.last_sim_result['simulated']['likely_failure_mode'][:12].replace(' ', '_')}"
        try:
            DigitalTwinService.save_scenario(
                machine_code=self.current_machine_code,
                name=scenario_name,
                baseline_state=self.last_sim_result["baseline"],
                simulated_parameters=self.last_sim_result["simulated"]["parameters"],
                predicted_risk_delta=self.last_sim_result["comparison"]["predicted_risk_delta_pct"],
                estimated_health_delta=self.last_sim_result["comparison"]["estimated_health_delta_pct"],
                notes=f"Simulated {self.last_sim_result['simulated']['likely_failure_mode']}",
            )
            QMessageBox.information(self, "Scenario Saved", f"Scenario '{scenario_name}' saved to SQLite database.")
            self.load_saved_scenarios()
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Scenario", str(e))

    def load_saved_scenarios(self) -> None:
        """Populate saved scenarios table from SQLite."""
        scenarios = DigitalTwinService.get_saved_scenarios(self.current_machine_code)
        self.tbl_scenarios.setRowCount(len(scenarios))
        for row, s in enumerate(scenarios):
            self.tbl_scenarios.setItem(row, 0, QTableWidgetItem(str(s["id"])))
            self.tbl_scenarios.setItem(row, 1, QTableWidgetItem(s["machine_code"]))
            self.tbl_scenarios.setItem(row, 2, QTableWidgetItem(s["name"]))
            r_item = QTableWidgetItem(f"{s['predicted_risk_delta']:+.1f}%")
            r_item.setForeground(Qt.yellow if s['predicted_risk_delta'] > 0 else Qt.cyan)
            self.tbl_scenarios.setItem(row, 3, r_item)
            h_item = QTableWidgetItem(f"{s['estimated_health_delta']:+.1f}%")
            self.tbl_scenarios.setItem(row, 4, h_item)
            self.tbl_scenarios.setItem(row, 5, QTableWidgetItem(s["created_at"]))
            self.tbl_scenarios.setItem(row, 6, QTableWidgetItem(s["notes"] or "-"))
