"""
Tests for IndusPredict AI Database Schema, Constraints, and Backup Utilities.
"""

from datetime import datetime
from pathlib import Path
import tempfile
import pytest
from sqlalchemy import inspect
from induspredict.db.models import (
    Base,
    Machine,
    MachineComponent,
    Role,
    Sensor,
    SensorReading,
    User,
    WorkOrder,
)
from induspredict.db.session import backup_database


def test_database_tables_exist(test_engine):
    """Verify that all required industrial database tables are created."""
    inspector = inspect(test_engine)
    table_names = set(inspector.get_table_names())

    required_tables = {
        "users",
        "roles",
        "machines",
        "machine_components",
        "sensors",
        "sensor_readings",
        "predictions",
        "rul_estimates",
        "anomaly_events",
        "model_versions",
        "model_evaluations",
        "drift_events",
        "work_orders",
        "maintenance_history",
        "simulation_scenarios",
        "audit_logs",
    }

    missing = required_tables - table_names
    assert not missing, f"Missing required database tables: {missing}"


def test_machine_sensor_relationship(test_engine, db_session):
    """Verify relational integrity between Machines, Components, and Sensors."""
    machine = db_session.query(Machine).filter(Machine.machine_code == "CMP-01").first()
    assert machine is not None
    assert len(machine.components) > 0
    assert len(machine.sensors) > 0

    # Test cascade constraint check
    sensor = machine.sensors[0]
    assert sensor.machine_id == machine.id
    assert sensor.sensor_code.startswith("CMP01")


def test_hot_backup_creation(test_engine, test_db_path):
    """Test hot database backup execution using SQLite online backup API."""
    tmp_backup_file = Path(tempfile.gettempdir()) / "test_hot_backup.db"
    if tmp_backup_file.exists():
        tmp_backup_file.unlink()

    # Monkeypatch session_module.DEFAULT_DB_FILE for test
    import induspredict.db.session as sm
    sm.DEFAULT_DB_FILE = test_db_path

    out_path = backup_database(destination_path=tmp_backup_file)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    tmp_backup_file.unlink()
