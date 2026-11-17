"""
IndusPredict AI - Industrial Dark UI Theme & Styling.
Implements dark navy and charcoal background, electric cyan and teal highlights,
modern card styling, and typography.
"""

# Color Palette Constants
BG_DARK = "#0D1117"        # Deepest dark navy/black
BG_PANEL = "#161B22"       # Dark charcoal panel background
BG_CARD = "#21262D"        # Slightly elevated card surface
BG_INPUT = "#0D1117"       # Text input background
BORDER_SUBTLE = "#30363D"  # Subtle divider borders
BORDER_FOCUS = "#58A6FF"   # Focus state border

# Accents
CYAN_PRIMARY = "#00F2FE"   # Electric Cyan
TEAL_ACCENT = "#00C9A7"    # Neon Industrial Teal
BLUE_ACCENT = "#4FACFE"    # Electric Blue

# Status Colors
COLOR_HEALTHY = "#2ECC71"  # Emerald Green
COLOR_WARNING = "#F39C12"  # Amber Orange
COLOR_CRITICAL = "#E74C3C" # Bright Crimson Red
COLOR_INFO = "#3498DB"     # Industrial Steel Blue

# Text Colors
TEXT_PRIMARY = "#F0F6FC"   # High contrast crisp white
TEXT_SECONDARY = "#8B949E" # Muted slate gray
TEXT_MUTED = "#6E7681"     # Darker caption gray

INDUSTRIAL_QSS = f"""
/* Universal Window & Font */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    font-size: 13px;
}}

/* Top Navigation Bar */
QFrame#topBar {{
    background-color: {BG_PANEL};
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding: 6px 16px;
}}

/* Sidebar */
QFrame#sideBar {{
    background-color: {BG_PANEL};
    border-right: 1px solid {BORDER_SUBTLE};
    min-width: 240px;
    max-width: 260px;
}}

/* Sidebar Navigation Buttons */
QPushButton.nav-btn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    text-align: left;
    padding: 10px 18px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}}

QPushButton.nav-btn:hover {{
    background-color: {BG_CARD};
    color: {CYAN_PRIMARY};
}}

QPushButton.nav-btn:checked, QPushButton.nav-btn.active {{
    background-color: rgba(0, 242, 254, 0.12);
    color: {CYAN_PRIMARY};
    border-left: 3px solid {CYAN_PRIMARY};
    font-weight: bold;
}}

/* Content Area */
QScrollArea {{
    background-color: {BG_DARK};
    border: none;
}}

/* Card Panels */
QFrame.card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 10px;
    padding: 14px;
}}

QFrame.card:hover {{
    border: 1px solid rgba(0, 242, 254, 0.4);
}}

/* Header Labels */
QLabel.page-title {{
    font-size: 22px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
}}

QLabel.page-subtitle {{
    font-size: 13px;
    color: {TEXT_SECONDARY};
}}

QLabel.card-title {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

QLabel.card-value {{
    font-size: 26px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
}}

/* Buttons */
QPushButton.primary-btn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {CYAN_PRIMARY}, stop:1 {BLUE_ACCENT});
    color: #050B14;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 13px;
}}

QPushButton.primary-btn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1DF6FF, stop:1 #67B6FF);
}}

QPushButton.secondary-btn {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}}

QPushButton.secondary-btn:hover {{
    background-color: {BORDER_SUBTLE};
    border-color: {BORDER_FOCUS};
}}

QPushButton.danger-btn {{
    background-color: rgba(231, 76, 60, 0.2);
    color: {COLOR_CRITICAL};
    border: 1px solid {COLOR_CRITICAL};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}}

QPushButton.danger-btn:hover {{
    background-color: {COLOR_CRITICAL};
    color: white;
}}

/* Inputs & Form Controls */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}}

QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
    border: 1px solid {CYAN_PRIMARY};
    background-color: #10151C;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    selection-background-color: rgba(0, 242, 254, 0.2);
    selection-color: {CYAN_PRIMARY};
    padding: 4px;
}}

/* Data Tables */
QTableWidget {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    gridline-color: {BORDER_SUBTLE};
    selection-background-color: rgba(0, 242, 254, 0.15);
    selection-color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid {BORDER_SUBTLE};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
}}

QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_SUBTLE};
    min-height: 20px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_SECONDARY};
}}

/* Badges */
QLabel.badge-healthy {{
    background-color: rgba(46, 204, 113, 0.15);
    color: {COLOR_HEALTHY};
    border: 1px solid {COLOR_HEALTHY};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}

QLabel.badge-warning {{
    background-color: rgba(243, 156, 18, 0.15);
    color: {COLOR_WARNING};
    border: 1px solid {COLOR_WARNING};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}

QLabel.badge-critical {{
    background-color: rgba(231, 76, 60, 0.15);
    color: {COLOR_CRITICAL};
    border: 1px solid {COLOR_CRITICAL};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}
"""
