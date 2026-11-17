"""
IndusPredict AI - Core Application Configuration.
Manages file system paths, database connection strings, security tokens,
and industrial telemetry thresholds.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

# Base Application Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DATA_DIR = Path(os.environ.get("INDUSPREDICT_DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = APP_DATA_DIR / "raw"
PROCESSED_DATA_DIR = APP_DATA_DIR / "processed"
MODELS_DIR = APP_DATA_DIR / "models"
BACKUP_DIR = APP_DATA_DIR / "backups"
REPORTS_DIR = APP_DATA_DIR / "reports"

# Ensure runtime directories exist
for p in [APP_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, BACKUP_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Safe cache dir for matplotlib in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "induspredict_mpl"))

# SQLite Database Configuration
DEFAULT_DB_FILE = APP_DATA_DIR / "induspredict.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_FILE}")

# Industrial Safety Constants
READ_ONLY_INDUSTRIAL_MODE = True  # Strict safeguard: Never emit control/write signals to PLCs/actuators

# Telemetry Configuration Defaults
DEFAULT_TELEMETRY_REFRESH_MS = 1000
MAX_LIVE_BUFFER_POINTS = 500

# Roles and Permissions
ROLE_ADMIN = "Administrator"
ROLE_ENGINEER = "Maintenance Engineer"
ROLE_RESEARCHER = "Researcher"
ROLE_VIEWER = "Viewer"

ALL_ROLES = [ROLE_ADMIN, ROLE_ENGINEER, ROLE_RESEARCHER, ROLE_VIEWER]

# Failure Risk Categories
RISK_NORMAL = "Normal"
RISK_WARNING = "Warning"
RISK_HIGH = "High Risk"
RISK_CRITICAL = "Critical"

ALL_RISK_CATEGORIES = [RISK_NORMAL, RISK_WARNING, RISK_HIGH, RISK_CRITICAL]

# Criticality Levels for Machines
CRITICALITY_LOW = "Low"
CRITICALITY_MEDIUM = "Medium"
CRITICALITY_HIGH = "High"
CRITICALITY_CRITICAL = "Mission Critical"

ALL_CRITICALITY_LEVELS = [CRITICALITY_LOW, CRITICALITY_MEDIUM, CRITICALITY_HIGH, CRITICALITY_CRITICAL]

# Standard Sensor Types & Measurement Units
SENSOR_SPECS = {
    "temperature": {"name": "Temperature", "unit": "°C", "min": 10.0, "max": 120.0, "warning": 85.0, "critical": 100.0},
    "vibration": {"name": "Vibration Velocity", "unit": "mm/s", "min": 0.0, "max": 25.0, "warning": 7.1, "critical": 11.2},
    "pressure": {"name": "Operating Pressure", "unit": "bar", "min": 0.0, "max": 30.0, "warning": 18.0, "critical": 24.0},
    "motor_speed": {"name": "Motor Speed", "unit": "RPM", "min": 0.0, "max": 4000.0, "warning": 3200.0, "critical": 3600.0},
    "current": {"name": "Motor Current", "unit": "A", "min": 0.0, "max": 150.0, "warning": 110.0, "critical": 130.0},
    "torque": {"name": "Torque", "unit": "Nm", "min": 0.0, "max": 350.0, "warning": 260.0, "critical": 300.0},
    "operating_hours": {"name": "Operating Hours", "unit": "hrs", "min": 0.0, "max": 50000.0, "warning": 30000.0, "critical": 40000.0},
}


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "IndusPredict AI"
    version: str = "1.0.0"
    organization: str = "Industrial AI & Cyber-Physical Systems Lab"
    database_url: str = DATABASE_URL
    db_file_path: Path = DEFAULT_DB_FILE
    read_only_mode: bool = READ_ONLY_INDUSTRIAL_MODE
    session_timeout_minutes: int = 120
