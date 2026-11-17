"""
Tests for IndusPredict AI Telemetry Service.
"""

from datetime import datetime, timedelta
import pytest
from induspredict.core.utils import utc_now
from induspredict.db.models import AnomalyEvent, Machine, Sensor, SensorReading
from induspredict.services.telemetry_service import TelemetryService


def test_telemetry_ingest_and_buffers(test_engine, db_session):
    """Verify live sliding buffer updates and SQLite batch persistence."""
    telemetry_svc = TelemetryService()

    frame = {
        "machine_code": "CMP-01",
        "timestamp": utc_now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "simulated",
        "sensors": {
            "temperature": {"value": 72.5},
            "vibration": {"value": 3.4},
        },
    }

    telemetry_svc.ingest_frame(frame)

    buf_temp = telemetry_svc.get_live_buffer("CMP-01", "temperature")
    buf_vib = telemetry_svc.get_live_buffer("CMP-01", "vibration")

    assert len(buf_temp) >= 1
    assert len(buf_vib) >= 1
    assert buf_temp[-1][1] == 72.5
    assert buf_vib[-1][1] == 3.4


def test_threshold_violation_triggers_anomaly(test_engine, db_session):
    """Verify that exceeding critical threshold produces an AnomalyEvent."""
    telemetry_svc = TelemetryService()

    # Query sensor critical threshold for CMP-01 temperature
    sensor = (
        db_session.query(Sensor)
        .join(Machine)
        .filter(Machine.machine_code == "CMP-01", Sensor.sensor_type == "temperature")
        .first()
    )
    assert sensor is not None
    m_id = sensor.machine_id
    crit_th = sensor.critical_threshold

    detected_alerts = []
    telemetry_svc.anomaly_detected.connect(lambda a: detected_alerts.append(a))

    # Ingest extreme temperature reading above critical threshold (e.g. 115°C)
    critical_frame = {
        "machine_code": "CMP-01",
        "timestamp": utc_now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "simulated",
        "sensors": {
            "temperature": {"value": crit_th + 5.0},
        },
    }
    telemetry_svc.ingest_frame(critical_frame)

    assert len(detected_alerts) >= 1
    alert = detected_alerts[-1]
    assert alert["machine_code"] == "CMP-01"
    assert alert["severity"] == "Critical"

    # Check database AnomalyEvent created
    anom = (
        db_session.query(AnomalyEvent)
        .filter(AnomalyEvent.machine_id == m_id, AnomalyEvent.severity == "Critical")
        .order_by(AnomalyEvent.timestamp.desc())
        .first()
    )
    assert anom is not None
    assert "exceeded" in anom.description


def test_sensor_connectivity_check(test_engine):
    """Verify detection of active, delayed, and disconnected sensor streams."""
    telemetry_svc = TelemetryService()

    now = utc_now()
    telemetry_svc.last_seen[("CMP-01", "temperature")] = now  # Just received
    telemetry_svc.last_seen[("CMP-01", "vibration")] = now - timedelta(seconds=12)  # Delayed
    telemetry_svc.last_seen[("CMP-01", "pressure")] = now - timedelta(seconds=30)   # Disconnected

    statuses = telemetry_svc.check_sensor_connectivity(timeout_seconds=8.0)
    assert statuses.get("CMP-01_temperature") == "ACTIVE"
    assert statuses.get("CMP-01_vibration") == "DELAYED"
    assert statuses.get("CMP-01_pressure") == "DISCONNECTED"


def test_update_sensor_thresholds(test_engine, db_session):
    """Verify updating and persisting sensor operational threshold limits."""
    sensor = db_session.query(Sensor).filter(Sensor.sensor_code == "CMP01-TEMP").first()
    assert sensor is not None
    sensor_id = sensor.id

    # Valid threshold update
    res = TelemetryService.update_sensor_thresholds(
        sensor_id=sensor_id,
        warning_val=88.0,
        critical_val=105.0,
    )
    assert res["success"] is True

    # Re-query in a fresh session
    updated_sensor = db_session.query(Sensor).filter(Sensor.id == sensor_id).first()
    db_session.refresh(updated_sensor)
    assert updated_sensor.warning_threshold == 88.0
    assert updated_sensor.critical_threshold == 105.0

    # Invalid: warning >= critical
    res_inv = TelemetryService.update_sensor_thresholds(
        sensor_id=sensor_id,
        warning_val=110.0,
        critical_val=100.0,
    )
    assert res_inv["success"] is False
    assert "strictly lower" in res_inv["error"]


def test_get_historical_sensor_data(test_engine):
    """Verify historical sensor queries from SQLite."""
    readings = TelemetryService.get_historical_sensor_data(machine_code="CMP-01", hours=48, limit=50)
    assert len(readings) > 0
    for r in readings:
        assert "timestamp" in r
        assert "value" in r
        assert "source" in r
        assert r["source"] in ("simulated", "uploaded", "mqtt", "opcua")
