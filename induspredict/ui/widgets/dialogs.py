"""
IndusPredict AI - Modal Dialogs.
Includes confirmation dialogs and machine creation/editing forms with validation.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from induspredict.core.config import ALL_CRITICALITY_LEVELS, CRITICALITY_MEDIUM
from induspredict.ui.theme import (
    BORDER_SUBTLE,
    COLOR_CRITICAL,
    CYAN_PRIMARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class ConfirmationDialog(QDialog):
    """Modern dark confirmation dialog for destructive or critical actions."""

    def __init__(self, title: str, message: str, is_danger: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLOR_CRITICAL if is_danger else CYAN_PRIMARY};")
        layout.addWidget(self.lbl_title)

        self.lbl_msg = QLabel(message)
        self.lbl_msg.setWordWrap(True)
        self.lbl_msg.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; line-height: 1.4;")
        layout.addWidget(self.lbl_msg)

        btn_box = QHBoxLayout()
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "secondary-btn")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_confirm = QPushButton("Confirm" if not is_danger else "Delete Permanently")
        if is_danger:
            self.btn_confirm.setProperty("class", "danger-btn")
        else:
            self.btn_confirm.setProperty("class", "primary-btn")
        self.btn_confirm.clicked.connect(self.accept)

        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_confirm)
        layout.addLayout(btn_box)


class MachineFormDialog(QDialog):
    """Modal form dialog for registering or modifying industrial equipment."""

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.is_edit = initial_data is not None
        self.setWindowTitle("Edit Machine Configuration" if self.is_edit else "Register New Industrial Asset")
        self.setModal(True)
        self.setMinimumWidth(480)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        title_lbl = QLabel(self.windowTitle())
        title_lbl.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {CYAN_PRIMARY};")
        main_layout.addWidget(title_lbl)

        form = QFormLayout()
        form.setSpacing(12)

        # Machine Code
        self.txt_code = QLineEdit()
        self.txt_code.setPlaceholderText("e.g. CMP-02, MTR-05")
        if self.is_edit:
            self.txt_code.setText(initial_data.get("code", ""))
            self.txt_code.setEnabled(False)  # Primary machine code is immutable
        form.addRow("Machine Code*:", self.txt_code)

        # Machine Name
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("e.g. Ingersoll Rand Nirvana Screw Compressor")
        if self.is_edit:
            self.txt_name.setText(initial_data.get("name", ""))
        form.addRow("Asset Name*:", self.txt_name)

        # Machine Type
        self.cb_type = QComboBox()
        self.cb_type.setEditable(True)
        self.cb_type.addItems([
            "Screw Compressor",
            "Induction Motor",
            "Centrifugal Pump",
            "Industrial Robot",
            "CNC Machining Center",
            "Hydraulic Press",
            "Turbine Generator",
        ])
        if self.is_edit:
            self.cb_type.setCurrentText(initial_data.get("type", "Screw Compressor"))
        form.addRow("Machine Type*:", self.cb_type)

        # Manufacturer
        self.txt_mfg = QLineEdit()
        self.txt_mfg.setPlaceholderText("e.g. Atlas Copco, Siemens, Sulzer")
        if self.is_edit:
            self.txt_mfg.setText(initial_data.get("manufacturer", ""))
        form.addRow("Manufacturer*:", self.txt_mfg)

        # Model Number
        self.txt_model = QLineEdit()
        if self.is_edit:
            self.txt_model.setText(initial_data.get("model", ""))
        form.addRow("Model Number:", self.txt_model)

        # Serial Number
        self.txt_serial = QLineEdit()
        if self.is_edit:
            self.txt_serial.setText(initial_data.get("serial", ""))
        form.addRow("Serial Number:", self.txt_serial)

        # Location
        self.txt_location = QLineEdit()
        self.txt_location.setPlaceholderText("e.g. Plant 1 - Bay B, Utility Line 4")
        if self.is_edit:
            self.txt_location.setText(initial_data.get("location", ""))
        form.addRow("Location*:", self.txt_location)

        # Criticality
        self.cb_crit = QComboBox()
        self.cb_crit.addItems(ALL_CRITICALITY_LEVELS)
        if self.is_edit:
            self.cb_crit.setCurrentText(initial_data.get("criticality", CRITICALITY_MEDIUM))
        else:
            self.cb_crit.setCurrentText(CRITICALITY_MEDIUM)
        form.addRow("Criticality Level:", self.cb_crit)

        # Operating Hours
        self.spin_hours = QDoubleSpinBox()
        self.spin_hours.setRange(0.0, 1000000.0)
        self.spin_hours.setDecimals(1)
        self.spin_hours.setSuffix(" hrs")
        if self.is_edit:
            self.spin_hours.setValue(float(initial_data.get("operating_hours", 0.0)))
        form.addRow("Operating Hours:", self.spin_hours)

        # Description / Notes
        self.txt_desc = QTextEdit()
        self.txt_desc.setMaximumHeight(70)
        if self.is_edit:
            self.txt_desc.setText(initial_data.get("description", ""))
        form.addRow("Description:", self.txt_desc)

        main_layout.addLayout(form)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "secondary-btn")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_save = QPushButton("Save Machine" if self.is_edit else "Register Machine")
        self.btn_save.setProperty("class", "primary-btn")
        self.btn_save.clicked.connect(self._validate_and_accept)

        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_save)
        main_layout.addLayout(btn_box)

    def _validate_and_accept(self) -> None:
        """Validate required fields before submitting."""
        code = self.txt_code.text().strip()
        name = self.txt_name.text().strip()
        mfg = self.txt_mfg.text().strip()
        loc = self.txt_location.text().strip()

        if not code or len(code) < 2:
            QMessageBox.warning(self, "Validation Error", "Machine Code must be at least 2 characters.")
            self.txt_code.setFocus()
            return

        if not name or len(name) < 3:
            QMessageBox.warning(self, "Validation Error", "Asset Name must be at least 3 characters.")
            self.txt_name.setFocus()
            return

        if not mfg:
            QMessageBox.warning(self, "Validation Error", "Manufacturer is required.")
            self.txt_mfg.setFocus()
            return

        if not loc:
            QMessageBox.warning(self, "Validation Error", "Plant Location is required.")
            self.txt_location.setFocus()
            return

        self.accept()

    def get_data(self) -> Dict[str, Any]:
        """Return validated dictionary of form entries."""
        return {
            "code": self.txt_code.text().strip().upper(),
            "name": self.txt_name.text().strip(),
            "type": self.cb_type.currentText().strip(),
            "manufacturer": self.txt_mfg.text().strip(),
            "model": self.txt_model.text().strip(),
            "serial": self.txt_serial.text().strip(),
            "location": self.txt_location.text().strip(),
            "criticality": self.cb_crit.currentText(),
            "operating_hours": self.spin_hours.value(),
            "description": self.txt_desc.toPlainText().strip(),
            "installation_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }
