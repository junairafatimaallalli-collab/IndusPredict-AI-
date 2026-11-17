"""
IndusPredict AI - Login & Session Initialization View.
Features professional industrial authentication, password hashing,
role-based session handling, error diagnostics, and administrative user registration.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from induspredict.core.config import ALL_ROLES, ROLE_VIEWER
from induspredict.core.security import SessionManager, UserSession
from induspredict.services.auth_service import AuthService
from induspredict.ui.theme import (
    BG_CARD,
    BG_PANEL,
    BORDER_SUBTLE,
    COLOR_CRITICAL,
    COLOR_HEALTHY,
    CYAN_PRIMARY,
    TEAL_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class RegisterDialog(QDialog):
    """Administrator modal for creating new user credentials with assigned roles."""

    def __init__(self, admin_session: Optional[UserSession] = None, parent=None):
        super().__init__(parent)
        self.admin_session = admin_session
        self.setWindowTitle("Register New Operator / Researcher")
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title_lbl = QLabel("User Registration")
        title_lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {CYAN_PRIMARY};")
        layout.addWidget(title_lbl)

        # Full Name
        self.txt_fullname = QLineEdit()
        self.txt_fullname.setPlaceholderText("Full Name (e.g. Dr. Jane Smith)")
        layout.addWidget(self.txt_fullname)

        # Username
        self.txt_username = QLineEdit()
        self.txt_username.setPlaceholderText("Username (e.g. jsmith)")
        layout.addWidget(self.txt_username)

        # Email
        self.txt_email = QLineEdit()
        self.txt_email.setPlaceholderText("Corporate Email (e.g. jsmith@plant.com)")
        layout.addWidget(self.txt_email)

        # Password
        self.txt_pwd = QLineEdit()
        self.txt_pwd.setEchoMode(QLineEdit.Password)
        self.txt_pwd.setPlaceholderText("Password (min 6 characters)")
        layout.addWidget(self.txt_pwd)

        # Role
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Assigned Role:"))
        self.cb_role = QComboBox()
        self.cb_role.addItems(ALL_ROLES)
        self.cb_role.setCurrentText(ROLE_VIEWER)
        role_layout.addWidget(self.cb_role)
        layout.addLayout(role_layout)

        # Status / error message
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color: {COLOR_CRITICAL}; font-size: 12px;")
        layout.addWidget(self.lbl_status)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "secondary-btn")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_create = QPushButton("Create Account")
        self.btn_create.setProperty("class", "primary-btn")
        self.btn_create.clicked.connect(self._handle_create)

        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_create)
        layout.addLayout(btn_box)

    def _handle_create(self) -> None:
        uname = self.txt_username.text().strip()
        email = self.txt_email.text().strip()
        pwd = self.txt_pwd.text()
        fname = self.txt_fullname.text().strip()
        role = self.cb_role.currentText()

        res = AuthService.register_user(
            username=uname,
            email=email,
            password=pwd,
            full_name=fname,
            role_name=role,
            admin_session=self.admin_session,
        )

        if res["success"]:
            QMessageBox.information(self, "Success", f"User account '{uname}' successfully registered.")
            self.accept()
        else:
            self.lbl_status.setText(res.get("error", "Registration failed."))


class LoginView(QWidget):
    """Main application login screen."""

    login_successful = Signal(object)  # Emits UserSession

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setAlignment(Qt.AlignCenter)

        # Login Center Card
        card = QFrame()
        card.setProperty("class", "card")
        card.setFixedWidth(420)
        card.setStyleSheet(f"""
            QFrame.card {{
                background-color: {BG_PANEL};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 12px;
                padding: 28px;
            }}
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(16)

        # Logo / Title
        header = QVBoxLayout()
        header.setAlignment(Qt.AlignCenter)
        header.setSpacing(4)

        logo_icon = QLabel("⚙️")
        logo_icon.setStyleSheet("font-size: 38px;")
        logo_icon.setAlignment(Qt.AlignCenter)
        header.addWidget(logo_icon)

        title_lbl = QLabel("IndusPredict AI")
        title_lbl.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {CYAN_PRIMARY};")
        title_lbl.setAlignment(Qt.AlignCenter)
        header.addWidget(title_lbl)

        sub_lbl = QLabel("Intelligent Industrial Predictive Maintenance")
        sub_lbl.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY};")
        sub_lbl.setAlignment(Qt.AlignCenter)
        header.addWidget(sub_lbl)

        card_layout.addLayout(header)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"border: none; border-bottom: 1px solid {BORDER_SUBTLE}; margin: 8px 0px;")
        card_layout.addWidget(div)

        # Username Input
        lbl_user = QLabel("Username or Operator ID")
        lbl_user.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.txt_username = QLineEdit()
        self.txt_username.setPlaceholderText("e.g. admin, engineer, researcher")
        self.txt_username.setText("admin")  # Pre-fill default for seamless evaluation
        card_layout.addWidget(lbl_user)
        card_layout.addWidget(self.txt_username)

        # Password Input
        lbl_pwd = QLabel("Security Password")
        lbl_pwd.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {TEXT_SECONDARY};")
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.Password)
        self.txt_password.setPlaceholderText("Enter password")
        self.txt_password.setText("Admin@123")  # Pre-fill default for seamless evaluation
        self.txt_password.returnPressed.connect(self._handle_login)
        card_layout.addWidget(lbl_pwd)
        card_layout.addWidget(self.txt_password)

        # Error Banner
        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet(f"color: {COLOR_CRITICAL}; font-size: 12px; font-weight: bold;")
        self.lbl_error.hide()
        card_layout.addWidget(self.lbl_error)

        # Sign In Button
        self.btn_login = QPushButton("Authenticate & Enter System")
        self.btn_login.setProperty("class", "primary-btn")
        self.btn_login.setFixedHeight(40)
        self.btn_login.clicked.connect(self._handle_login)
        card_layout.addWidget(self.btn_login)

        # Quick Role Selectors for Evaluation
        quick_box = QVBoxLayout()
        quick_box.setSpacing(6)
        quick_lbl = QLabel("Quick Evaluation Credentials:")
        quick_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        quick_box.addWidget(quick_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        roles_presets = [
            ("Admin", "admin", "Admin@123"),
            ("Engineer", "engineer", "Engineer@123"),
            ("Researcher", "researcher", "Researcher@123"),
            ("Viewer", "viewer", "Viewer@123"),
        ]

        for label, u, p in roles_presets:
            b = QPushButton(label)
            b.setProperty("class", "secondary-btn")
            b.setStyleSheet("font-size: 11px; padding: 4px 8px;")
            b.clicked.connect(lambda checked=False, un=u, pw=p: self._set_credentials(un, pw))
            btn_row.addWidget(b)

        quick_box.addLayout(btn_row)
        card_layout.addLayout(quick_box)

        # Register prompt
        reg_layout = QHBoxLayout()
        reg_layout.setAlignment(Qt.AlignCenter)
        lbl_new = QLabel("Need an account?")
        lbl_new.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        btn_reg = QPushButton("Register")
        btn_reg.setStyleSheet(f"background: none; border: none; color: {CYAN_PRIMARY}; font-weight: bold; font-size: 12px;")
        btn_reg.clicked.connect(self._handle_register)
        reg_layout.addWidget(lbl_new)
        reg_layout.addWidget(btn_reg)
        card_layout.addLayout(reg_layout)

        outer_layout.addWidget(card)

    def _set_credentials(self, u: str, p: str) -> None:
        self.txt_username.setText(u)
        self.txt_password.setText(p)
        self.lbl_error.hide()

    def _handle_login(self) -> None:
        username = self.txt_username.text().strip()
        password = self.txt_password.text()

        if not username or not password:
            self.lbl_error.setText("Please enter both username and password.")
            self.lbl_error.show()
            return

        session = AuthService.authenticate(username, password)
        if session:
            self.lbl_error.hide()
            self.login_successful.emit(session)
        else:
            self.lbl_error.setText("Authentication failed. Invalid username or security password.")
            self.lbl_error.show()

    def _handle_register(self) -> None:
        current_session = SessionManager.get_current_session()
        dialog = RegisterDialog(admin_session=current_session, parent=self)
        dialog.exec()
