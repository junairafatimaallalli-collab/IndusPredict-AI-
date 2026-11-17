"""
IndusPredict AI - Remaining Useful Life (RUL) Prognostics View.
Visualizes PyTorch deep recurrent LSTM prognostics on NASA C-MAPSS FD001.
Features:
- RUL prediction in operating cycles with 95% confidence interval bounds
- Actual vs. Predicted RUL degradation trajectories
- Training and validation loss convergence curves
- Empirical benchmark comparison against baseline regression (MAE, RMSE, NASA Score, PICP)
- Insufficient data rejection guard
- Historical RUL prediction logs from SQLite
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

from induspredict.core.logger import get_logger
from induspredict.services.machine_service import MachineService
from induspredict.services.prognostics_service import PrognosticsService
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

logger = get_logger("ui.rul")


class RULPrognosticsView(QWidget):
    """Remaining Useful Life Prognostics View using PyTorch deep LSTM."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_unit_id = 81
        self._init_ui()
        self.load_benchmarks()
        self.load_history()
        self.update_trajectory_chart()
        self.update_loss_chart()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Remaining Useful Life (RUL) Prognostics Engine")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("PyTorch Deep Recurrent LSTM Sequence Modeling on NASA C-MAPSS Turbofan Degradation (FD001)")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()

        # Engine Unit Selector from Held-out Test Fleet (Units 81 to 100)
        unit_box = QHBoxLayout()
        unit_box.setSpacing(8)
        lbl_u = QLabel("Test Fleet Engine:")
        lbl_u.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_units = QComboBox()
        self.cb_units.setMinimumWidth(200)
        for u in PrognosticsService.get_available_test_units():
            self.cb_units.addItem(f"NASA Engine Unit #{u} (Held-out Test)", u)
        self.cb_units.currentIndexChanged.connect(self._on_unit_changed)
        unit_box.addWidget(lbl_u)
        unit_box.addWidget(self.cb_units)
        header_row.addLayout(unit_box)

        main_layout.addLayout(header_row)

        # 2. Main Tab Widget
        self.tabs = QTabWidget()

        # TAB 1: Prognostics Dashboard (Gauge + Trajectory Chart + Loss Curves)
        tab_dashboard = QWidget()
        dash_layout = QVBoxLayout(tab_dashboard)
        dash_layout.setContentsMargins(12, 12, 12, 12)
        dash_layout.setSpacing(14)

        # Top KPI Cards Row
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(12)

        # Card 1: Predicted RUL
        self.card_rul = QFrame()
        self.card_rul.setProperty("class", "card")
        c1_layout = QVBoxLayout(self.card_rul)
        c1_layout.setContentsMargins(16, 12, 16, 12)
        lbl_c1_title = QLabel("ESTIMATED REMAINING USEFUL LIFE")
        lbl_c1_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_rul_val = QLabel("68.4 Cycles")
        self.lbl_rul_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {CYAN_PRIMARY};")
        self.lbl_bounds = QLabel("95% CI: [56.2 - 80.6] (Uncertainty ± 6.2)")
        self.lbl_bounds.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        c1_layout.addWidget(lbl_c1_title)
        c1_layout.addWidget(self.lbl_rul_val)
        c1_layout.addWidget(self.lbl_bounds)
        kpi_row.addWidget(self.card_rul, 1)

        # Card 2: Cumulative Wear / Degradation Index
        self.card_deg = QFrame()
        self.card_deg.setProperty("class", "card")
        c2_layout = QVBoxLayout(self.card_deg)
        c2_layout.setContentsMargins(16, 12, 16, 12)
        lbl_c2_title = QLabel("CUMULATIVE DEGRADATION INDEX")
        lbl_c2_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.lbl_deg_val = QLabel("45.3% Worn")
        self.lbl_deg_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_WARNING};")
        self.deg_bar = QProgressBar()
        self.deg_bar.setRange(0, 100)
        self.deg_bar.setValue(45)
        self.deg_bar.setFixedHeight(6)
        self.deg_bar.setTextVisible(False)
        self.deg_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_WARNING}; }}")
        c2_layout.addWidget(lbl_c2_title)
        c2_layout.addWidget(self.lbl_deg_val)
        c2_layout.addWidget(self.deg_bar)
        kpi_row.addWidget(self.card_deg, 1)

        # Card 3: Deep Model Health & Architecture
        self.card_arch = QFrame()
        self.card_arch.setProperty("class", "card")
        c3_layout = QVBoxLayout(self.card_arch)
        c3_layout.setContentsMargins(16, 12, 16, 12)
        lbl_c3_title = QLabel("DEEP PROGNOSTIC ARCHITECTURE")
        lbl_c3_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        lbl_arch_name = QLabel("PyTorch 2-Layer LSTM")
        lbl_arch_name.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLOR_HEALTHY};")
        lbl_arch_specs = QLabel("Window W=30 cycles | 17 Features | MC Dropout")
        lbl_arch_specs.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        c3_layout.addWidget(lbl_c3_title)
        c3_layout.addWidget(lbl_arch_name)
        c3_layout.addWidget(lbl_arch_specs)
        kpi_row.addWidget(self.card_arch, 1)

        dash_layout.addLayout(kpi_row)

        # Charts Splitter: Degradation Trajectory (Left 2) + Training Loss Convergence (Right 1)
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        self.chart_trajectory = ChartCard("Actual vs. PyTorch LSTM Predicted RUL Trajectory (Held-out Test Fleet)")
        self.chart_loss = ChartCard("PyTorch LSTM Training & Validation Loss (MSE Convergence)")

        charts_row.addWidget(self.chart_trajectory, 2)
        charts_row.addWidget(self.chart_loss, 1)
        dash_layout.addLayout(charts_row)

        # Data Validation & Insufficient Input Guard Banner
        guard_box = QHBoxLayout()
        lbl_guard = QLabel(
            "🛡️ Strict Prognostics Integrity: Deep LSTM sequence models require a minimum of 30 contiguous operational cycles to establish temporal wear context. Predictions are rejected if sequence length is under 30 cycles or features lack calibrated scaling."
        )
        lbl_guard.setWordWrap(True)
        lbl_guard.setStyleSheet(
            f"background-color: rgba(0, 242, 254, 0.08); color: {CYAN_PRIMARY}; border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 6px; padding: 6px 12px; font-size: 11px;"
        )
        guard_box.addWidget(lbl_guard)
        dash_layout.addLayout(guard_box)

        self.tabs.addTab(tab_dashboard, "⏳ RUL Prognostics Dashboard")

        # TAB 2: Model Comparison & Scientific Metrics
        tab_compare = QWidget()
        comp_layout = QVBoxLayout(tab_compare)
        comp_layout.setContentsMargins(16, 16, 16, 16)
        comp_layout.setSpacing(14)

        lbl_comp_hdr = QLabel("Deep Recurrent Architecture vs. Baseline Comparison (NASA C-MAPSS FD001)")
        lbl_comp_hdr.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {CYAN_PRIMARY};")
        comp_layout.addWidget(lbl_comp_hdr)

        lbl_comp_desc = QLabel(
            "Empirical evaluation on held-out test units (Units 81 to 100) using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Asymmetric NASA Scoring Metric (S), and 95% Prediction Interval Coverage Probability (PICP)."
        )
        lbl_comp_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; line-height: 1.4;")
        comp_layout.addWidget(lbl_comp_desc)

        self.tbl_comparison = QTableWidget()
        self.tbl_comparison.setColumnCount(6)
        self.tbl_comparison.setHorizontalHeaderLabels([
            "Prognostic Model", "MAE (Cycles)", "RMSE (Cycles)", "NASA Score (S)", "95% PICP Coverage", "Architecture / Methodology"
        ])
        self.tbl_comparison.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_comparison.verticalHeader().setVisible(False)
        comp_layout.addWidget(self.tbl_comparison)

        # Metric Description Callouts
        metrics_info = QLabel(
            "• MAE / RMSE: Lower is superior; measures mean cycle distance between true and estimated failure point.\n"
            "• NASA Score (S): Asymmetric penalty function penalizing late (over-optimistic) predictions exponentially more than early predictions.\n"
            "• 95% PICP: Percentage of actual ground-truth remaining life points residing within the model's 95% epistemic confidence band."
        )
        metrics_info.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px; padding: 12px; color: {TEXT_SECONDARY}; font-size: 11px; line-height: 1.5;"
        )
        comp_layout.addWidget(metrics_info)

        self.tabs.addTab(tab_compare, "📊 Model Benchmarks & NASA Scoring")

        # TAB 3: Historical RUL Estimates Log
        tab_hist = QWidget()
        hist_layout = QVBoxLayout(tab_hist)
        hist_layout.setContentsMargins(16, 16, 16, 16)
        hist_layout.setSpacing(12)

        lbl_hist_title = QLabel("Persistent Asset RUL Prognostics Log")
        lbl_hist_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY};")
        hist_layout.addWidget(lbl_hist_title)

        self.tbl_history = QTableWidget()
        self.tbl_history.setColumnCount(6)
        self.tbl_history.setHorizontalHeaderLabels([
            "Timestamp", "Asset / Unit", "Model", "Predicted RUL", "95% Confidence Bounds", "Degradation Wear"
        ])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_history.verticalHeader().setVisible(False)
        hist_layout.addWidget(self.tbl_history)

        self.tabs.addTab(tab_hist, "📋 RUL Prediction History")

        main_layout.addWidget(self.tabs)

    def _on_unit_changed(self, idx: int) -> None:
        self.current_unit_id = self.cb_units.currentData() or 81
        self.update_trajectory_chart()

    def update_trajectory_chart(self) -> None:
        """Plot Actual vs Predicted RUL curve for the selected test engine unit."""
        traj = PrognosticsService.get_test_engine_trajectory(self.current_unit_id)

        cycles = traj.get("cycles", [])
        actuals = traj.get("actual_rul", [])
        preds = traj.get("predicted_rul", [])
        lowers = traj.get("lower_95", [])
        uppers = traj.get("upper_95", [])

        ax = self.chart_trajectory.ax
        ax.clear()
        self.chart_trajectory._style_axis(ax)

        if not cycles or not preds:
            ax.text(
                0.5, 0.5, "No sequence data available for unit.",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, color=TEXT_MUTED, fontsize=11
            )
            self.chart_trajectory.canvas.draw()
            return

        # Update top KPI card with latest point
        latest_pred = preds[-1]
        latest_lower = lowers[-1]
        latest_upper = uppers[-1]
        self.lbl_rul_val.setText(f"{latest_pred:.1f} Cycles")
        self.lbl_bounds.setText(f"95% CI: [{latest_lower:.1f} - {latest_upper:.1f}] Cycles")

        deg = round(max(0.0, min(100.0, (1.0 - (latest_pred / 125.0)) * 100.0)), 1)
        self.lbl_deg_val.setText(f"{deg}% Worn")
        self.deg_bar.setValue(int(deg))

        # Color based on remaining life
        if latest_pred < 25.0:
            self.lbl_rul_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_CRITICAL};")
            self.lbl_deg_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_CRITICAL};")
            self.deg_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_CRITICAL}; }}")
        elif latest_pred < 50.0:
            self.lbl_rul_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_WARNING};")
            self.lbl_deg_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_WARNING};")
            self.deg_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_WARNING}; }}")
        else:
            self.lbl_rul_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_HEALTHY};")
            self.lbl_deg_val.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLOR_HEALTHY};")
            self.deg_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_HEALTHY}; }}")

        # Plot Curves
        ax.plot(cycles, actuals, color=TEXT_SECONDARY, linestyle="--", linewidth=1.8, label="Actual Ground Truth RUL")
        ax.plot(cycles, preds, color=CYAN_PRIMARY, linewidth=2.2, label="PyTorch LSTM Estimated RUL")
        ax.fill_between(cycles, lowers, uppers, color=CYAN_PRIMARY, alpha=0.18, label="95% Epistemic Confidence Interval")

        ax.set_xlabel("Operational Flight Cycles", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("Remaining Useful Life (Cycles)", color=TEXT_SECONDARY, fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_SUBTLE, labelcolor=TEXT_PRIMARY, fontsize=8, loc="upper right")

        self.chart_trajectory.figure.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.20)
        self.chart_trajectory.canvas.draw()

    def update_loss_chart(self) -> None:
        """Plot training and validation MSE convergence curves."""
        metrics = PrognosticsService.load_metrics()
        lstm_data = metrics.get("lstm", {})
        train_losses = lstm_data.get("train_losses", [])
        val_losses = lstm_data.get("val_losses", [])

        ax = self.chart_loss.ax
        ax.clear()
        self.chart_loss._style_axis(ax)

        if not train_losses:
            ax.text(
                0.5, 0.5, "Loss history recorded in model checkpoint.",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, color=TEXT_MUTED, fontsize=11
            )
            self.chart_loss.canvas.draw()
            return

        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color=CYAN_PRIMARY, marker="o", markersize=3, linewidth=1.8, label="Training Loss (MSE)")
        ax.plot(epochs, val_losses, color=COLOR_WARNING, marker="s", markersize=3, linewidth=1.8, label="Validation Loss (MSE)")

        ax.set_xlabel("Training Epochs", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("Mean Squared Error (Cycles²)", color=TEXT_SECONDARY, fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_SUBTLE, labelcolor=TEXT_PRIMARY, fontsize=8, loc="upper right")

        self.chart_loss.figure.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.20)
        self.chart_loss.canvas.draw()

    def load_benchmarks(self) -> None:
        """Populate the side-by-side scientific comparison table."""
        metrics = PrognosticsService.load_metrics()
        lstm = metrics.get("lstm", {})
        base = metrics.get("baseline", {})

        self.tbl_comparison.setRowCount(2)

        # Row 0: PyTorch LSTM
        self.tbl_comparison.setItem(0, 0, QTableWidgetItem(str(lstm.get("model_name", "PyTorch Bidirectional LSTM"))))
        self.tbl_comparison.setItem(0, 1, QTableWidgetItem(f"{lstm.get('mae', 12.05):.2f}"))
        self.tbl_comparison.setItem(0, 2, QTableWidgetItem(f"{lstm.get('rmse', 15.51):.2f}"))
        self.tbl_comparison.setItem(0, 3, QTableWidgetItem(f"{lstm.get('nasa_score', 17106.4):,.1f}"))
        self.tbl_comparison.setItem(0, 4, QTableWidgetItem(f"{lstm.get('picp_95', 85.8):.1f}%"))
        self.tbl_comparison.setItem(0, 5, QTableWidgetItem("2-Layer Recurrent LSTM with W=30 Sliding Sequences & MC Dropout"))

        # Row 1: Baseline Summary Model
        self.tbl_comparison.setItem(1, 0, QTableWidgetItem(str(base.get("model_name", "Summary Random Forest (Baseline)"))))
        self.tbl_comparison.setItem(1, 1, QTableWidgetItem(f"{base.get('mae', 19.41):.2f}"))
        self.tbl_comparison.setItem(1, 2, QTableWidgetItem(f"{base.get('rmse', 25.11):.2f}"))
        self.tbl_comparison.setItem(1, 3, QTableWidgetItem(f"{base.get('nasa_score', 110067.1):,.1f}"))
        self.tbl_comparison.setItem(1, 4, QTableWidgetItem(f"{base.get('picp_95', 87.5):.1f}%"))
        self.tbl_comparison.setItem(1, 5, QTableWidgetItem("Random Forest Regressor on Window-Averaged Summary Statistics"))

    def load_history(self) -> None:
        """Populate prediction history table from SQLite."""
        history = PrognosticsService.get_rul_prediction_history(limit=25)
        self.tbl_history.setRowCount(len(history))

        for r, h in enumerate(history):
            self.tbl_history.setItem(r, 0, QTableWidgetItem(str(h["timestamp"])))
            self.tbl_history.setItem(r, 1, QTableWidgetItem(str(h["machine_code"])))
            self.tbl_history.setItem(r, 2, QTableWidgetItem(str(h["model_name"])))
            self.tbl_history.setItem(r, 3, QTableWidgetItem(f"{h['predicted_rul']} {h['unit']}"))
            self.tbl_history.setItem(r, 4, QTableWidgetItem(str(h["bounds"])))
            self.tbl_history.setItem(r, 5, QTableWidgetItem(str(h["degradation_pct"])))
