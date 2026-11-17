"""
IndusPredict AI - Explainable AI (XAI) View (Page 8).
Visualizes game-theoretic SHAP feature attributions, local prediction waterfalls,
global model importance rankings, normal vs abnormal telemetry comparisons,
and natural language engineering advisories with explicit physical disclaimers.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from induspredict.ml.explainability import SHAPExplainer
from induspredict.services.machine_service import MachineService
from induspredict.services.xai_service import XAIService
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

logger = get_logger("ui.xai")


class XAIView(QWidget):
    """Page 8: Explainable AI (SHAP) View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_model = "Random Forest"
        self._init_ui()
        self.load_global_importance()
        self.run_sample_explanation()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Explainable Artificial Intelligence (SHAP)")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Game-theoretic Shapley feature attributions, local waterfall diagnostics, and engineering narratives")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()

        # Model Selector Dropdown
        m_box = QHBoxLayout()
        m_box.setSpacing(8)
        lbl_m = QLabel("Explaining Model:")
        lbl_m.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_models = QComboBox()
        self.cb_models.setMinimumWidth(180)
        for m in ["Random Forest", "XGBoost", "LightGBM"]:
            self.cb_models.addItem(m)
        self.cb_models.currentTextChanged.connect(self._on_model_changed)
        m_box.addWidget(lbl_m)
        m_box.addWidget(self.cb_models)
        header_row.addLayout(m_box)

        main_layout.addLayout(header_row)

        # Disclaimer Banner (Scientific Integrity)
        banner_frame = QFrame()
        banner_frame.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 6px; padding: 4px;"
        )
        b_lay = QHBoxLayout(banner_frame)
        b_lay.setContentsMargins(10, 6, 10, 6)
        lbl_disc = QLabel(f"⚠️ {SHAPExplainer.DISCLAIMER}")
        lbl_disc.setWordWrap(True)
        lbl_disc.setStyleSheet(f"color: {CYAN_PRIMARY}; font-size: 11px;")
        b_lay.addWidget(lbl_disc)
        main_layout.addWidget(banner_frame)

        # Main Tabs Widget
        self.tabs = QTabWidget()

        # TAB 1: Local Prediction Explanation (Waterfall / Contributions)
        tab_local = QWidget()
        loc_layout = QVBoxLayout(tab_local)
        loc_layout.setContentsMargins(12, 12, 12, 12)
        loc_layout.setSpacing(12)

        # Top Preset Stress Row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        lbl_pr = QLabel("Evaluate Stress Condition:")
        lbl_pr.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        preset_row.addWidget(lbl_pr)

        btn_nom = QPushButton("Nominal Operating")
        btn_nom.clicked.connect(self._preset_nominal)
        preset_row.addWidget(btn_nom)

        btn_therm = QPushButton("Thermal Overload")
        btn_therm.clicked.connect(self._preset_thermal)
        preset_row.addWidget(btn_therm)

        btn_strain = QPushButton("Overstrain & Wear")
        btn_strain.clicked.connect(self._preset_strain)
        preset_row.addWidget(btn_strain)

        preset_row.addStretch()
        loc_layout.addLayout(preset_row)

        # Splitter: Left chart (Waterfall/Attribution Bar) + Right Narrative & Breakdown
        split_h = QHBoxLayout()
        split_h.setSpacing(12)

        # Local Attribution Chart
        self.chart_local = ChartCard("Local SHAP Feature Attributions (φᵢ Impact on Failure Probability)")
        split_h.addWidget(self.chart_local, 2)

        # Right Card: Engineering Narrative & Normal/Abnormal Comparison
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Narrative Card
        card_narrative = QFrame()
        card_narrative.setProperty("class", "card")
        n_lay = QVBoxLayout(card_narrative)
        n_lay.setContentsMargins(14, 12, 14, 12)
        lbl_n_title = QLabel("MAINTENANCE ADVISORY & ROOT DRIVER ANALYSIS")
        lbl_n_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {CYAN_PRIMARY};")
        self.txt_narrative = QTextEdit()
        self.txt_narrative.setReadOnly(True)
        self.txt_narrative.setFixedHeight(120)
        self.txt_narrative.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER_SUBTLE}; color: {TEXT_PRIMARY}; font-size: 12px; line-height: 1.4;"
        )
        n_lay.addWidget(lbl_n_title)
        n_lay.addWidget(self.txt_narrative)
        right_panel.addWidget(card_narrative)

        # Normal vs Abnormal Table
        card_comp = QFrame()
        card_comp.setProperty("class", "card")
        c_lay = QVBoxLayout(card_comp)
        c_lay.setContentsMargins(14, 12, 14, 12)
        lbl_c_title = QLabel("TELEMETRY vs. NOMINAL BASELINE BENCHMARKS")
        lbl_c_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {CYAN_PRIMARY};")
        self.tbl_comparison = QTableWidget()
        self.tbl_comparison.setColumnCount(4)
        self.tbl_comparison.setHorizontalHeaderLabels(["Sensor / Parameter", "Actual", "Nominal Range", "Status"])
        self.tbl_comparison.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_comparison.verticalHeader().setVisible(False)
        self.tbl_comparison.setFixedHeight(150)
        c_lay.addWidget(lbl_c_title)
        c_lay.addWidget(self.tbl_comparison)
        right_panel.addWidget(card_comp)

        split_h.addLayout(right_panel, 1)
        loc_layout.addLayout(split_h)

        self.tabs.addTab(tab_local, "🔍 Local Prediction Explanation")

        # TAB 2: Global Feature Importance (Population Level)
        tab_global = QWidget()
        glob_layout = QVBoxLayout(tab_global)
        glob_layout.setContentsMargins(16, 16, 16, 16)
        glob_layout.setSpacing(14)

        lbl_g_hdr = QLabel("Global SHAP Feature Importance Rankings")
        lbl_g_hdr.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {CYAN_PRIMARY};")
        glob_layout.addWidget(lbl_g_hdr)

        lbl_g_desc = QLabel(
            "Mean absolute Shapley values E[|φ|] computed across the reference industrial fleet (UCI AI4I 2020). "
            "Higher values denote parameters with the greatest aggregate predictive power across all failure modes."
        )
        lbl_g_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        glob_layout.addWidget(lbl_g_desc)

        glob_split = QHBoxLayout()
        glob_split.setSpacing(12)

        self.chart_global = ChartCard("Global Mean |SHAP| Feature Impact Ranking")
        glob_split.addWidget(self.chart_global, 2)

        self.tbl_global = QTableWidget()
        self.tbl_global.setColumnCount(4)
        self.tbl_global.setHorizontalHeaderLabels(["Rank", "Feature Parameter", "Mean |SHAP|", "Global Impact %"])
        self.tbl_global.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_global.verticalHeader().setVisible(False)
        glob_split.addWidget(self.tbl_global, 1)

        glob_layout.addLayout(glob_split)
        self.tabs.addTab(tab_global, "🌐 Global Model Feature Importance")

        main_layout.addWidget(self.tabs)

    def _on_model_changed(self, text: str) -> None:
        if text:
            self.current_model = text
            self.load_global_importance()
            self.run_sample_explanation()

    def run_sample_explanation(self, sensor_dict: Optional[Dict[str, float]] = None) -> None:
        """Execute SHAP local attribution on the provided or default parameters."""
        if sensor_dict is None:
            # High overstrain default
            sensor_dict = {
                "air_temperature": 25.0,
                "process_temperature": 35.0,
                "rotational_speed": 1320.0,
                "torque": 68.0,
                "tool_wear": 180.0,
                "machine_type": "M",
            }

        try:
            exp = XAIService.explain_active_prediction(sensor_dict, model_name=self.current_model)

            # Update Narrative
            self.txt_narrative.setText(exp["engineering_summary"])

            # Update Comparison Table
            comp = exp["normal_vs_abnormal"]
            self.tbl_comparison.setRowCount(len(comp))
            for row, (k, v) in enumerate(comp.items()):
                self.tbl_comparison.setItem(row, 0, QTableWidgetItem(v["metric"]))
                self.tbl_comparison.setItem(row, 1, QTableWidgetItem(str(v["actual"])))
                self.tbl_comparison.setItem(row, 2, QTableWidgetItem(v["nominal_range"]))
                st_item = QTableWidgetItem(v["status"])
                st_item.setForeground(Qt.red if v["is_abnormal"] else Qt.green)
                self.tbl_comparison.setItem(row, 3, st_item)

            # Render Local Attribution Chart
            self._render_local_chart(exp["attributions"])

        except Exception as e:
            logger.error(f"Error executing SHAP explanation: {e}")

    def _render_local_chart(self, attributions: List[Dict[str, Any]]) -> None:
        """Render horizontal diverging bar chart representing SHAP values."""
        self.chart_local.clear()
        ax = self.chart_local.get_axes()

        top_attrs = attributions[:7]
        top_attrs.reverse()

        names = [a["display_name"] for a in top_attrs]
        vals = [a["shap_value"] for a in top_attrs]
        colors = [COLOR_CRITICAL if v > 0 else COLOR_HEALTHY for v in vals]

        y_pos = range(len(names))
        ax.barh(y_pos, vals, color=colors, height=0.6, alpha=0.85)
        ax.axvline(0, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("Shapley Attribution Value φ (Impact on Failure Probability)", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.2)

        self.chart_local.draw()

    def load_global_importance(self) -> None:
        """Compute and render global feature importance."""
        try:
            rankings = XAIService.get_global_importance(model_name=self.current_model)

            # Update Table
            self.tbl_global.setRowCount(len(rankings))
            for row, r in enumerate(rankings):
                self.tbl_global.setItem(row, 0, QTableWidgetItem(f"#{row + 1}"))
                self.tbl_global.setItem(row, 1, QTableWidgetItem(r["display_name"]))
                self.tbl_global.setItem(row, 2, QTableWidgetItem(f"{r['mean_abs_shap']:.4f}"))
                self.tbl_global.setItem(row, 3, QTableWidgetItem(f"{r['global_importance_pct']:.1f}%"))

            # Render Bar Chart
            self.chart_global.clear()
            ax = self.chart_global.get_axes()

            top = rankings[:8]
            top.reverse()
            names = [r["display_name"] for r in top]
            vals = [r["global_importance_pct"] for r in top]

            y_pos = range(len(names))
            ax.barh(y_pos, vals, color=CYAN_PRIMARY, height=0.6, alpha=0.85)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names, fontsize=9, color=TEXT_PRIMARY)
            ax.set_xlabel("Global Predictive Importance (%)", color=TEXT_SECONDARY, fontsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.2)

            self.chart_global.draw()
        except Exception as e:
            logger.error(f"Error loading global importance: {e}")

    def _preset_nominal(self) -> None:
        self.run_sample_explanation({
            "air_temperature": 25.0,
            "process_temperature": 35.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 20.0,
            "machine_type": "M",
        })

    def _preset_thermal(self) -> None:
        self.run_sample_explanation({
            "air_temperature": 35.0,
            "process_temperature": 48.0,
            "rotational_speed": 1350.0,
            "torque": 45.0,
            "tool_wear": 50.0,
            "machine_type": "M",
        })

    def _preset_strain(self) -> None:
        self.run_sample_explanation({
            "air_temperature": 25.0,
            "process_temperature": 35.0,
            "rotational_speed": 1320.0,
            "torque": 68.0,
            "tool_wear": 180.0,
            "machine_type": "M",
        })
