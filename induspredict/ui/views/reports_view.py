"""
IndusPredict AI - Reports and Analytics View (Page 10).
Provides enterprise multi-format reporting (PDF via ReportLab, Excel via XlsxWriter, CSV)
across Machine Health, Failure Predictions, RUL Prognostics, Sensor Telemetry,
Maintenance Work Orders, and Research Model Benchmarks.
"""

from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.logger import get_logger
from induspredict.services.machine_service import MachineService
from induspredict.services.report_service import ReportService
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

logger = get_logger("ui.reports")


class ReportsView(QWidget):
    """Page 10: Reports & Analytics View."""

    REPORT_TYPES = [
        "Machine Health & Asset Inventory",
        "Failure Risk & Anomaly Predictions",
        "Remaining Useful Life (RUL) Prognostics",
        "Maintenance Work Orders & History",
        "Model Benchmarks & Research Evaluations",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_report_data: Optional[Dict[str, Any]] = None
        self._init_ui()
        self.load_report_preview()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Certified Reports & Scientific Analytics")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Export ISO 13374 compliant PDF reports, multi-sheet Excel workbooks, and CSV analytics logs")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)

        header_row.addStretch()
        main_layout.addLayout(header_row)

        # 2. Controls & Filters Ribbon Card
        ribbon = QFrame()
        ribbon.setProperty("class", "card")
        r_layout = QHBoxLayout(ribbon)
        r_layout.setContentsMargins(14, 12, 14, 12)
        r_layout.setSpacing(12)

        # Report Type Selector
        t_box = QVBoxLayout()
        t_box.setSpacing(4)
        lbl_rt = QLabel("Report Category:")
        lbl_rt.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_report_type = QComboBox()
        self.cb_report_type.setMinimumWidth(260)
        for rt in self.REPORT_TYPES:
            self.cb_report_type.addItem(rt)
        self.cb_report_type.currentIndexChanged.connect(lambda _: self.load_report_preview())
        t_box.addWidget(lbl_rt)
        t_box.addWidget(self.cb_report_type)
        r_layout.addLayout(t_box)

        # Machine Selector
        m_box = QVBoxLayout()
        m_box.setSpacing(4)
        lbl_m = QLabel("Machine Filter:")
        lbl_m.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.cb_machine = QComboBox()
        self.cb_machine.setMinimumWidth(180)
        self.cb_machine.addItem("All Machines (Fleet-Wide)", "All")
        for m in MachineService.get_all_machines():
            self.cb_machine.addItem(f"{m['code']} - {m['name'][:12]}", m["code"])
        self.cb_machine.currentIndexChanged.connect(lambda _: self.load_report_preview())
        m_box.addWidget(lbl_m)
        m_box.addWidget(self.cb_machine)
        r_layout.addLayout(m_box)

        r_layout.addStretch()

        # Action Export Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.btn_export_pdf = QPushButton("📄 Export Certified PDF")
        self.btn_export_pdf.setProperty("class", "btn-primary")
        self.btn_export_pdf.setFixedHeight(34)
        self.btn_export_pdf.clicked.connect(self._export_pdf)
        btn_box.addWidget(self.btn_export_pdf)

        self.btn_export_excel = QPushButton("📊 Export Excel (.xlsx)")
        self.btn_export_excel.setFixedHeight(34)
        self.btn_export_excel.clicked.connect(self._export_excel)
        btn_box.addWidget(self.btn_export_excel)

        self.btn_export_csv = QPushButton("📈 Export CSV")
        self.btn_export_csv.setFixedHeight(34)
        self.btn_export_csv.clicked.connect(self._export_csv)
        btn_box.addWidget(self.btn_export_csv)

        r_layout.addLayout(btn_box)
        main_layout.addWidget(ribbon)

        # 3. Dynamic Summary KPI Strip
        self.kpi_strip = QHBoxLayout()
        self.kpi_strip.setSpacing(12)

        self.kpi_cards = []
        for i in range(4):
            card = QFrame()
            card.setProperty("class", "card")
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(12, 10, 12, 10)
            c_lay.setSpacing(2)
            lbl_k_title = QLabel(f"METRIC {i+1}")
            lbl_k_title.setStyleSheet(f"font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase;")
            lbl_k_val = QLabel("-")
            lbl_k_val.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {CYAN_PRIMARY};")
            c_lay.addWidget(lbl_k_title)
            c_lay.addWidget(lbl_k_val)
            self.kpi_strip.addWidget(card, 1)
            self.kpi_cards.append((lbl_k_title, lbl_k_val))

        main_layout.addLayout(self.kpi_strip)

        # 4. Preview Table Card
        preview_card = QFrame()
        preview_card.setProperty("class", "card")
        p_layout = QVBoxLayout(preview_card)
        p_layout.setContentsMargins(14, 12, 14, 12)
        p_layout.setSpacing(10)

        self.lbl_table_hdr = QLabel("Report Data Preview (Live SQLite Table)")
        self.lbl_table_hdr.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {CYAN_PRIMARY};")
        p_layout.addWidget(self.lbl_table_hdr)

        self.tbl_preview = QTableWidget()
        self.tbl_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_preview.verticalHeader().setVisible(False)
        self.tbl_preview.setSelectionBehavior(QTableWidget.SelectRows)
        p_layout.addWidget(self.tbl_preview)

        main_layout.addWidget(preview_card, 1)

    def load_report_preview(self) -> None:
        """Fetch and populate preview table and KPI summary cards."""
        rep_type = self.cb_report_type.currentText()
        m_code = self.cb_machine.currentData() or "All"

        try:
            data = ReportService.get_report_data(rep_type, machine_code=m_code)
            self.current_report_data = data

            # Update KPI strip
            kpis = list(data["summary_kpis"].items())
            for idx in range(4):
                title_lbl, val_lbl = self.kpi_cards[idx]
                if idx < len(kpis):
                    title_lbl.setText(kpis[idx][0].replace("_", " ").title())
                    val_lbl.setText(str(kpis[idx][1]))
                else:
                    title_lbl.setText("-")
                    val_lbl.setText("-")

            # Update Table Preview
            cols = data["table_columns"]
            rows = data["table_rows"]
            self.tbl_preview.setColumnCount(len(cols))
            self.tbl_preview.setHorizontalHeaderLabels(cols)
            self.tbl_preview.setRowCount(len(rows))

            for r_idx, row in enumerate(rows):
                for c_idx, cell in enumerate(row):
                    item = QTableWidgetItem(str(cell))
                    if str(cell) in ["Critical", "Emergency", "High Risk"]:
                        item.setForeground(Qt.red)
                    elif str(cell) in ["Normal", "Healthy", "Completed"]:
                        item.setForeground(Qt.green)
                    elif str(cell) in ["Warning", "Medium"]:
                        item.setForeground(Qt.yellow)
                    self.tbl_preview.setItem(r_idx, c_idx, item)

            self.lbl_table_hdr.setText(f"Report Data Preview: {rep_type} ({len(rows)} Records)")

        except Exception as e:
            logger.error(f"Error loading report preview: {e}")

    def _export_pdf(self) -> None:
        rep_type = self.cb_report_type.currentText()
        m_code = self.cb_machine.currentData() or "All"
        safe_name = f"IndusPredict_{rep_type[:15].replace(' ', '_')}_{m_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Certified PDF Report", safe_name, "PDF Documents (*.pdf)")
        if file_path:
            try:
                out = ReportService.export_pdf(rep_type, m_code, Path(file_path))
                QMessageBox.information(self, "PDF Generated", f"Certified audit report successfully saved to:\n{out}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to generate PDF: {e}")

    def _export_excel(self) -> None:
        rep_type = self.cb_report_type.currentText()
        m_code = self.cb_machine.currentData() or "All"
        safe_name = f"IndusPredict_{rep_type[:15].replace(' ', '_')}_{m_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel Workbook", safe_name, "Excel Workbooks (*.xlsx)")
        if file_path:
            try:
                out = ReportService.export_excel(rep_type, m_code, Path(file_path))
                QMessageBox.information(self, "Excel Generated", f"Formatted workbook successfully saved to:\n{out}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to generate Excel workbook: {e}")

    def _export_csv(self) -> None:
        rep_type = self.cb_report_type.currentText()
        m_code = self.cb_machine.currentData() or "All"
        safe_name = f"IndusPredict_{rep_type[:15].replace(' ', '_')}_{m_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV Dataset", safe_name, "CSV Files (*.csv)")
        if file_path:
            try:
                out = ReportService.export_csv(rep_type, m_code, Path(file_path))
                QMessageBox.information(self, "CSV Generated", f"Standardized CSV file successfully saved to:\n{out}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to generate CSV: {e}")
