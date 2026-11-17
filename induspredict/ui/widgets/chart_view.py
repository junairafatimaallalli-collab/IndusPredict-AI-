"""
IndusPredict AI - Embedded Industrial Chart View.
Embeds Matplotlib with a dark industrial cyber-physical theme into PySide6.
"""

from typing import Dict, List, Optional
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from induspredict.ui.theme import (
    BG_CARD,
    BG_PANEL,
    BORDER_SUBTLE,
    COLOR_CRITICAL,
    COLOR_HEALTHY,
    COLOR_WARNING,
    CYAN_PRIMARY,
    TEAL_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class ChartCard(QFrame):
    """Container card wrapping a Matplotlib canvas with dark aesthetics."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {TEXT_PRIMARY}; text-transform: uppercase; letter-spacing: 0.5px;"
        )
        layout.addWidget(self.title_label)

        # Figure & Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor(BG_CARD)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {BG_CARD};")
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self._style_axis(self.ax)

    def clear(self) -> None:
        """Clear axes and re-apply dark theme."""
        self.ax.clear()
        self._style_axis(self.ax)

    def get_axes(self):
        """Return the matplotlib axes."""
        return self.ax

    def draw(self) -> None:
        """Redraw canvas with margin padding."""
        self.figure.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.20)
        self.canvas.draw()

    def _style_axis(self, ax) -> None:
        """Apply dark theme colors to matplotlib axes."""
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER_SUBTLE)
        ax.grid(True, linestyle="--", alpha=0.3, color=BORDER_SUBTLE)

    def plot_health_trends(self, timestamps: List[str], values: List[float], anomalies: Optional[List[int]] = None) -> None:
        """Plot continuous machine health trajectory and anomaly indicators."""
        self.ax.clear()
        self._style_axis(self.ax)

        if not timestamps or not values:
            self.ax.text(
                0.5, 0.5, "Awaiting Telemetry Ingestion...",
                horizontalalignment="center", verticalalignment="center",
                transform=self.ax.transAxes, color=TEXT_MUTED, fontsize=11
            )
            self.canvas.draw()
            return

        x_indices = np.arange(len(values))
        self.ax.plot(x_indices, values, color=CYAN_PRIMARY, linewidth=2, label="Health Metric")
        self.ax.fill_between(x_indices, values, alpha=0.15, color=CYAN_PRIMARY)

        # Mark anomaly exceedances
        if anomalies and any(anomalies):
            anomaly_indices = [i for i, a in enumerate(anomalies) if a > 0]
            anomaly_vals = [values[i] for i in anomaly_indices]
            self.ax.scatter(
                anomaly_indices, anomaly_vals,
                color=COLOR_CRITICAL, s=35, zorder=5, label="Anomaly Trigger"
            )

        # Show sparse x-ticks to prevent overcrowding
        step = max(1, len(timestamps) // 6)
        tick_pos = list(range(0, len(timestamps), step))
        self.ax.set_xticks(tick_pos)
        self.ax.set_xticklabels([timestamps[i] for i in tick_pos], rotation=25, ha="right", color=TEXT_SECONDARY)
        self.ax.set_ylabel("Reading / Health Index", color=TEXT_SECONDARY, fontsize=9)
        self.ax.legend(facecolor=BG_CARD, edgecolor=BORDER_SUBTLE, labelcolor=TEXT_PRIMARY, fontsize=8, loc="upper right")

        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.25)
        self.canvas.draw()

    def plot_risk_distribution(self, distribution: Dict[str, int]) -> None:
        """Plot categorical failure risk distribution bars."""
        self.ax.clear()
        self._style_axis(self.ax)

        categories = ["Normal", "Warning", "High Risk", "Critical"]
        counts = [distribution.get(c, 0) for c in categories]
        colors = [COLOR_HEALTHY, COLOR_WARNING, "#FF7A00", COLOR_CRITICAL]

        bars = self.ax.bar(categories, counts, color=colors, width=0.55, edgecolor=BORDER_SUBTLE)
        self.ax.set_ylabel("Asset Count", color=TEXT_SECONDARY, fontsize=9)

        # Value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=9, fontweight="bold"
            )

        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.25)
        self.canvas.draw()
