"""
IndusPredict AI - Metric Stat Card Widget.
Reusable industrial dashboard card displaying a KPI value, title, icon, and status badge.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from induspredict.ui.theme import (
    BG_CARD,
    BORDER_SUBTLE,
    CYAN_PRIMARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class StatCard(QFrame):
    """High-contrast modern industrial KPI card."""

    def __init__(
        self,
        title: str,
        value: str,
        icon_symbol: str = "📊",
        subtitle: Optional[str] = None,
        accent_color: str = CYAN_PRIMARY,
        parent=None,
    ):
        super().__init__(parent)
        self.accent_color = accent_color
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(105)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # Header row: Title + Icon
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setProperty("class", "card-title")
        self.lbl_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 600; font-size: 11px; letter-spacing: 0.8px;")

        self.lbl_icon = QLabel(icon_symbol)
        self.lbl_icon.setStyleSheet(f"font-size: 18px; color: {accent_color};")

        header_row.addWidget(self.lbl_title)
        header_row.addStretch()
        header_row.addWidget(self.lbl_icon)

        # Value row
        self.lbl_value = QLabel(str(value))
        self.lbl_value.setProperty("class", "card-value")
        self.lbl_value.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {accent_color};")

        layout.addLayout(header_row)
        layout.addWidget(self.lbl_value)

        # Subtitle / delta if provided
        if subtitle:
            self.lbl_sub = QLabel(subtitle)
            self.lbl_sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
            layout.addWidget(self.lbl_sub)
        else:
            self.lbl_sub = None

    def update_value(self, new_val: Any, new_sub: Optional[str] = None) -> None:
        """Dynamically update card value and optional subtitle."""
        self.lbl_value.setText(str(new_val))
        if new_sub and self.lbl_sub:
            self.lbl_sub.setText(new_sub)
