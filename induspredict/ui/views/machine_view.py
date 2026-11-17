"""
IndusPredict AI - Machine Management View.
Complete asset lifecycle management: CRUD operations, component hierarchies,
sensor binding configurations, maintenance history, and search/filter toolbars.
"""

from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.config import ALL_CRITICALITY_LEVELS
from induspredict.core.security import (
    PERM_MACHINE_DELETE,
    PERM_MACHINE_EDIT,
    SessionManager,
)
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
from induspredict.ui.widgets.dialogs import ConfirmationDialog, MachineFormDialog


class MachineView(QWidget):
    """Industrial asset management screen providing full SQLite CRUD and detail inspection."""

    machine_modified = Signal()  # Emitted when a machine is added, edited, or removed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_selected_machine_id: Optional[int] = None
        self._init_ui()
        self.load_machines()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header & Actions Toolbar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Industrial Machine Registry & Configuration")
        lbl_title.setProperty("class", "page-title")
        lbl_sub = QLabel("Manage cyber-physical equipment profiles, installed sensor hardware, and component lifespans")
        lbl_sub.setProperty("class", "page-subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        top_bar.addLayout(title_box)

        top_bar.addStretch()

        # Action Buttons
        self.btn_add = QPushButton("+ Register New Machine")
        self.btn_add.setProperty("class", "primary-btn")
        self.btn_add.clicked.connect(self._handle_add_machine)
        top_bar.addWidget(self.btn_add)

        self.btn_edit = QPushButton("✏️ Edit Details")
        self.btn_edit.setProperty("class", "secondary-btn")
        self.btn_edit.setEnabled(False)
        self.btn_edit.clicked.connect(self._handle_edit_machine)
        top_bar.addWidget(self.btn_edit)

        self.btn_delete = QPushButton("🗑️ Delete Machine")
        self.btn_delete.setProperty("class", "danger-btn")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._handle_delete_machine)
        top_bar.addWidget(self.btn_delete)

        main_layout.addLayout(top_bar)

        # 2. Search & Filter Bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(10)

        # Search Input
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search by code, name, manufacturer, or location...")
        self.txt_search.textChanged.connect(self.load_machines)
        filter_bar.addWidget(self.txt_search, 2)

        # Type Filter
        lbl_type = QLabel("Type:")
        lbl_type.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.cb_type_filter = QComboBox()
        self.cb_type_filter.addItems(["All", "Screw Compressor", "Induction Motor", "Centrifugal Pump", "Industrial Robot", "CNC Machining Center"])
        self.cb_type_filter.currentIndexChanged.connect(self.load_machines)
        filter_bar.addWidget(lbl_type)
        filter_bar.addWidget(self.cb_type_filter)

        # Criticality Filter
        lbl_crit = QLabel("Criticality:")
        lbl_crit.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.cb_crit_filter = QComboBox()
        self.cb_crit_filter.addItems(["All"] + ALL_CRITICALITY_LEVELS)
        self.cb_crit_filter.currentIndexChanged.connect(self.load_machines)
        filter_bar.addWidget(lbl_crit)
        filter_bar.addWidget(self.cb_crit_filter)

        # Refresh
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setProperty("class", "secondary-btn")
        self.btn_refresh.setFixedWidth(36)
        self.btn_refresh.clicked.connect(self.load_machines)
        filter_bar.addWidget(self.btn_refresh)

        main_layout.addLayout(filter_bar)

        # 3. Splitter: Machine Table (Top) and Detailed Inspection Tabs (Bottom)
        splitter = QSplitter(Qt.Vertical)

        # Machines Table
        table_container = QFrame()
        table_container.setProperty("class", "card")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(8, 8, 8, 8)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "ID", "Code", "Asset Name", "Type", "Manufacturer", "Location", "Hours", "Criticality", "Status", "Sensors"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_container)

        # Details Inspection Panel (Tabs)
        self.details_container = QFrame()
        self.details_container.setProperty("class", "card")
        details_layout = QVBoxLayout(self.details_container)
        details_layout.setContentsMargins(12, 10, 12, 10)

        self.lbl_selected_asset = QLabel("Select an industrial machine above to inspect specifications & sensors")
        self.lbl_selected_asset.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {CYAN_PRIMARY}; margin-bottom: 6px;")
        details_layout.addWidget(self.lbl_selected_asset)

        self.tabs = QTabWidget()

        # Tab 1: Components
        self.tbl_components = QTableWidget()
        self.tbl_components.setColumnCount(6)
        self.tbl_components.setHorizontalHeaderLabels(["ID", "Component Name", "Type", "Part Number", "Hours", "Status"])
        self.tbl_components.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_components.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_components, "Installed Components")

        # Tab 2: Sensors Configuration
        self.tbl_sensors = QTableWidget()
        self.tbl_sensors.setColumnCount(7)
        self.tbl_sensors.setHorizontalHeaderLabels(["Code", "Sensor Name", "Type", "Unit", "Range", "Warning Thresh", "Status"])
        self.tbl_sensors.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_sensors.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_sensors, "Sensor Telemetry Configuration")

        # Tab 3: Maintenance History
        self.tbl_history = QTableWidget()
        self.tbl_history.setColumnCount(5)
        self.tbl_history.setHorizontalHeaderLabels(["Service Date", "Action Taken", "Parts Replaced", "Downtime", "Technician"])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_history.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_history, "Maintenance Service Log")

        details_layout.addWidget(self.tabs)
        splitter.addWidget(self.details_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)

    def load_machines(self) -> None:
        """Fetch machines matching current search and filter settings."""
        search = self.txt_search.text().strip()
        t_filter = self.cb_type_filter.currentText()
        c_filter = self.cb_crit_filter.currentText()

        machines = MachineService.get_all_machines(
            type_filter=t_filter,
            criticality_filter=c_filter,
            search_query=search,
        )

        self.table.setRowCount(len(machines))
        for row, m in enumerate(machines):
            id_item = QTableWidgetItem(str(m["id"]))
            id_item.setData(Qt.UserRole, m["id"])
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QTableWidgetItem(str(m["code"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(m["name"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(m["type"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(m["manufacturer"])))
            self.table.setItem(row, 5, QTableWidgetItem(str(m["location"])))
            self.table.setItem(row, 6, QTableWidgetItem(f"{m['operating_hours']:.1f} hrs"))

            # Criticality Badge
            crit_item = QTableWidgetItem(str(m["criticality"]))
            if m["criticality"] in ("High", "Mission Critical"):
                crit_item.setForeground(Qt.yellow)
            self.table.setItem(row, 7, crit_item)

            # Status Badge
            stat_item = QTableWidgetItem(str(m["status"]).upper())
            if m["status"] == "healthy":
                stat_item.setForeground(Qt.green)
            elif m["status"] == "warning":
                stat_item.setForeground(Qt.yellow)
            else:
                stat_item.setForeground(Qt.red)
            self.table.setItem(row, 8, stat_item)

            self.table.setItem(row, 9, QTableWidgetItem(f"{m['sensors_count']} Active"))

        # Maintain selection if possible
        if self.current_selected_machine_id:
            for r in range(self.table.rowCount()):
                item = self.table.item(r, 0)
                if item and item.data(Qt.UserRole) == self.current_selected_machine_id:
                    self.table.selectRow(r)
                    break
        elif self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _on_table_selection_changed(self) -> None:
        """Handle row selection to load sub-components and sensor details."""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            self.btn_edit.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.current_selected_machine_id = None
            self.lbl_selected_asset.setText("Select an industrial machine above to inspect specifications & sensors")
            return

        row = self.table.currentRow()
        id_item = self.table.item(row, 0)
        if not id_item:
            return

        machine_id = id_item.data(Qt.UserRole)
        self.current_selected_machine_id = machine_id

        # RBAC button states
        session = SessionManager.get_current_session()
        can_edit = session.has_permission(PERM_MACHINE_EDIT) if session else True
        can_delete = session.has_permission(PERM_MACHINE_DELETE) if session else True

        self.btn_edit.setEnabled(can_edit)
        self.btn_delete.setEnabled(can_delete)

        # Load details
        details = MachineService.get_machine_details(machine_id)
        if not details:
            return

        self.lbl_selected_asset.setText(
            f"Asset: [{details['code']}] {details['name']} | Mfg: {details['manufacturer']} | Loc: {details['location']} | Hours: {details['operating_hours']:.1f} hrs"
        )

        # 1. Populate Components
        comps = details["components"]
        self.tbl_components.setRowCount(len(comps))
        for r, c in enumerate(comps):
            self.tbl_components.setItem(r, 0, QTableWidgetItem(str(c["id"])))
            self.tbl_components.setItem(r, 1, QTableWidgetItem(str(c["name"])))
            self.tbl_components.setItem(r, 2, QTableWidgetItem(str(c["type"])))
            self.tbl_components.setItem(r, 3, QTableWidgetItem(str(c["part_number"] or "Standard")))
            self.tbl_components.setItem(r, 4, QTableWidgetItem(f"{c['current_operating_hours']:.0f} / {c['expected_lifespan_hours']:.0f} hrs ({c['wear_pct']}%)"))
            self.tbl_components.setItem(r, 5, QTableWidgetItem(str(c["status"]).title()))

        # 2. Populate Sensors
        sensors = details["sensors"]
        self.tbl_sensors.setRowCount(len(sensors))
        for r, s in enumerate(sensors):
            self.tbl_sensors.setItem(r, 0, QTableWidgetItem(str(s["code"])))
            self.tbl_sensors.setItem(r, 1, QTableWidgetItem(str(s["name"])))
            self.tbl_sensors.setItem(r, 2, QTableWidgetItem(str(s["type"])))
            self.tbl_sensors.setItem(r, 3, QTableWidgetItem(str(s["unit"])))
            self.tbl_sensors.setItem(r, 4, QTableWidgetItem(f"{s['min_threshold']} - {s['max_threshold']} {s['unit']}"))
            self.tbl_sensors.setItem(r, 5, QTableWidgetItem(f"{s['warning_threshold']} {s['unit']}"))
            self.tbl_sensors.setItem(r, 6, QTableWidgetItem(str(s["status"]).upper()))

        # 3. Populate Maintenance History
        records = details["maintenance_records"]
        self.tbl_history.setRowCount(len(records))
        for r, h in enumerate(records):
            self.tbl_history.setItem(r, 0, QTableWidgetItem(str(h["date"])))
            self.tbl_history.setItem(r, 1, QTableWidgetItem(str(h["action"])))
            self.tbl_history.setItem(r, 2, QTableWidgetItem(str(h["parts"])))
            self.tbl_history.setItem(r, 3, QTableWidgetItem(f"{h['downtime_hours']} hrs"))
            self.tbl_history.setItem(r, 4, QTableWidgetItem(str(h["technician"])))

    def _handle_add_machine(self) -> None:
        """Launch machine creation dialog and persist to SQLite."""
        session = SessionManager.get_current_session()
        dialog = MachineFormDialog(parent=self)
        if dialog.exec():
            data = dialog.get_data()
            res = MachineService.create_machine(data, user_session=session)
            if res["success"]:
                QMessageBox.information(self, "Success", f"Machine '{res['code']}' registered successfully.")
                self.load_machines()
                self.machine_modified.emit()
            else:
                QMessageBox.critical(self, "Error", res.get("error", "Failed to register machine."))

    def _handle_edit_machine(self) -> None:
        """Launch editing form for the currently selected machine."""
        if not self.current_selected_machine_id:
            return

        session = SessionManager.get_current_session()
        details = MachineService.get_machine_details(self.current_selected_machine_id)
        if not details:
            return

        dialog = MachineFormDialog(initial_data=details, parent=self)
        if dialog.exec():
            data = dialog.get_data()
            res = MachineService.update_machine(self.current_selected_machine_id, data, user_session=session)
            if res["success"]:
                QMessageBox.information(self, "Success", "Machine updated successfully.")
                self.load_machines()
                self.machine_modified.emit()
            else:
                QMessageBox.critical(self, "Error", res.get("error", "Failed to update machine."))

    def _handle_delete_machine(self) -> None:
        """Confirm and permanently delete machine from database."""
        if not self.current_selected_machine_id:
            return

        session = SessionManager.get_current_session()
        details = MachineService.get_machine_details(self.current_selected_machine_id)
        if not details:
            return

        dialog = ConfirmationDialog(
            title="Confirm Machine Removal",
            message=f"Are you sure you want to permanently delete machine '{details['code']}' ({details['name']})?\n\nThis will remove all associated sensors, telemetry logs, predictions, and work orders.",
            is_danger=True,
            parent=self,
        )

        if dialog.exec():
            res = MachineService.delete_machine(self.current_selected_machine_id, user_session=session)
            if res["success"]:
                QMessageBox.information(self, "Deleted", f"Machine '{res['deleted_code']}' permanently removed.")
                self.current_selected_machine_id = None
                self.load_machines()
                self.machine_modified.emit()
            else:
                QMessageBox.critical(self, "Error", res.get("error", "Failed to delete machine."))
