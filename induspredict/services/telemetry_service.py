"""
IndusPredict AI - Telemetry Service.
Coordinates live in-memory sliding buffers, threshold monitoring,
dynamic anomaly event generation, sensor connection state tracking, and historical queries.
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QObject, Signal
from sqlalchemy.orm import selectinload

from induspredict.core.config import MAX_LIVE_BUFFER_POINTS, SENSOR_SPECS
from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import AnomalyEvent, Machine, Sensor, SensorReading
from induspredict.db.session import get_db_session

logger = get_logger("services.telemetry")


class TelemetryService(QObject):
    """
    Central telemetry coordinator.
    Receives readings from MQTT, OPC UA, CSV Uploader, or Simulator,
    updates ring buffers, monitors thresholds, triggers alerts, and persists records.
    """

    # Signals
    reading_processed = Signal(dict)   # Emitted for live UI update
    anomaly_detected = Signal(dict)    # Emitted when warning or critical threshold is violated
    sensor_status_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ring buffers: key = (machine_code, sensor_type), value = deque of (timestamp_str, float_val)
        self.live_buffers: Dict[tuple, deque] = {}
        # Last seen timestamp: key = (machine_code, sensor_type), value = datetime
        self.last_seen: Dict[tuple, datetime] = {}
        # Unpersisted buffer for batched SQLite writes
        self._db_batch: List[SensorReading] = []
        self._batch_threshold = 25

    def get_live_buffer(self, machine_code: str, sensor_type: str) -> List[Tuple[str, float]]:
        """Return list of (timestamp_str, value) points from in-memory ring buffer."""
        key = (machine_code, sensor_type)
        if key not in self.live_buffers:
            self.live_buffers[key] = deque(maxlen=MAX_LIVE_BUFFER_POINTS)
        return list(self.live_buffers[key])

    def ingest_frame(self, frame: Dict[str, Any]) -> None:
        """
        Process an incoming multi-sensor telemetry frame from any source.
        Frame format:
        {
            "machine_code": "CMP-01",
            "timestamp": "2026-11-17 19:00:00",
            "source": "mqtt" | "opcua" | "uploaded" | "simulated",
            "sensors": {
                "temperature": {"value": 78.5, ...},
                "vibration": {"value": 4.2, ...}, ...
            }
        }
        """
        machine_code = frame.get("machine_code")
        ts_str = frame.get("timestamp", utc_now().strftime("%Y-%m-%d %H:%M:%S"))
        source = frame.get("source", "simulated")
        sensors = frame.get("sensors", {})

        if not machine_code or not sensors:
            return

        now = utc_now()

        # Update in-memory buffers and check thresholds
        for stype, sdata in sensors.items():
            val = float(sdata.get("value", 0.0))
            key = (machine_code, stype)

            if key not in self.live_buffers:
                self.live_buffers[key] = deque(maxlen=MAX_LIVE_BUFFER_POINTS)

            self.live_buffers[key].append((ts_str[-8:], val))  # Store HH:MM:SS
            self.last_seen[key] = now

        # Threshold checking & SQLite record building
        with get_db_session() as session:
            m = session.query(Machine).options(selectinload(Machine.sensors)).filter(Machine.machine_code == machine_code).first()
            if not m:
                return

            sensors_by_type = {s.sensor_type: s for s in m.sensors}
            machine_status = m.status
            has_critical = False
            has_warning = False

            for stype, sdata in sensors.items():
                val = float(sdata.get("value", 0.0))
                sensor = sensors_by_type.get(stype)

                if sensor:
                    is_crit = val >= sensor.critical_threshold
                    is_warn = val >= sensor.warning_threshold

                    if is_crit:
                        has_critical = True
                        sensor.status = "error"
                    elif is_warn:
                        has_warning = True
                        sensor.status = "warning"
                    else:
                        sensor.status = "active"

                    reading = SensorReading(
                        machine_id=m.id,
                        sensor_id=sensor.id,
                        timestamp=now,
                        value=round(val, 2),
                        source=source,  # Strictly identified
                        is_anomaly=is_crit or is_warn,
                        quality_flag="GOOD",
                    )
                    self._db_batch.append(reading)

                    # Create AnomalyEvent on exceedance
                    if is_crit or is_warn:
                        sev = "Critical" if is_crit else "Warning"
                        anom = AnomalyEvent(
                            machine_id=m.id,
                            sensor_id=sensor.id,
                            timestamp=now,
                            anomaly_score=round(val / max(1.0, sensor.critical_threshold), 2),
                            severity=sev,
                            description=f"{sensor.name} reading {val:.2f} {sensor.unit} exceeded {sev.lower()} threshold ({sensor.critical_threshold if is_crit else sensor.warning_threshold} {sensor.unit}).",
                            acknowledged=False,
                        )
                        session.add(anom)

                        alert_payload = {
                            "machine_code": machine_code,
                            "sensor_name": sensor.name,
                            "sensor_type": stype,
                            "value": val,
                            "unit": sensor.unit,
                            "severity": sev,
                            "timestamp": ts_str,
                            "source": source,
                        }
                        self.anomaly_detected.emit(alert_payload)

            # Update overall machine status
            if has_critical:
                m.status = "critical"
            elif has_warning and m.status != "critical":
                m.status = "warning"

            # Flush batched readings
            if len(self._db_batch) >= self._batch_threshold:
                session.bulk_save_objects(self._db_batch)
                self._db_batch.clear()

            session.commit()

        self.reading_processed.emit(frame)

    def check_sensor_connectivity(self, timeout_seconds: float = 8.0) -> Dict[str, str]:
        """
        Check health and responsiveness of all known machine sensors.
        Returns Dict[(machine_code, sensor_type), status ('ACTIVE', 'DELAYED', 'DISCONNECTED')]
        """
        now = utc_now()
        statuses = {}
        for (m_code, stype), last_ts in self.last_seen.items():
            delta = (now - last_ts).total_seconds()
            if delta < timeout_seconds:
                statuses[f"{m_code}_{stype}"] = "ACTIVE"
            elif delta < timeout_seconds * 2:
                statuses[f"{m_code}_{stype}"] = "DELAYED"
            else:
                statuses[f"{m_code}_{stype}"] = "DISCONNECTED"
        return statuses

    @staticmethod
    def update_sensor_thresholds(
        sensor_id: int,
        warning_val: float,
        critical_val: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Configure and persist new operational threshold limits for a sensor."""
        if warning_val >= critical_val:
            return {"success": False, "error": "Warning threshold must be strictly lower than Critical threshold."}

        with get_db_session() as session:
            sensor = session.query(Sensor).filter(Sensor.id == sensor_id).first()
            if not sensor:
                return {"success": False, "error": f"Sensor ID {sensor_id} not found."}

            sensor.warning_threshold = float(warning_val)
            sensor.critical_threshold = float(critical_val)
            if min_val is not None:
                sensor.min_threshold = float(min_val)
            if max_val is not None:
                sensor.max_threshold = float(max_val)

            session.commit()
            logger.info(f"Sensor '{sensor.sensor_code}' thresholds updated: Warning={warning_val}, Critical={critical_val}")
            return {"success": True, "sensor_code": sensor.sensor_code}

    @staticmethod
    def get_historical_sensor_data(
        machine_code: str,
        sensor_type: Optional[str] = None,
        hours: int = 24,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        """Retrieve chronological historical sensor records from SQLite."""
        with get_db_session() as session:
            cutoff = utc_now() - timedelta(hours=hours)
            query = (
                session.query(SensorReading)
                .join(Machine)
                .join(Sensor)
                .filter(Machine.machine_code == machine_code, SensorReading.timestamp >= cutoff)
            )

            if sensor_type and sensor_type != "All":
                query = query.filter(Sensor.sensor_type == sensor_type)

            readings = query.order_by(SensorReading.timestamp.desc()).limit(limit).all()

            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "sensor_code": r.sensor.sensor_code,
                    "sensor_name": r.sensor.name,
                    "sensor_type": r.sensor.sensor_type,
                    "value": r.value,
                    "unit": r.sensor.unit,
                    "source": r.source,  # Strictly identified
                    "is_anomaly": r.is_anomaly,
                    "quality_flag": r.quality_flag,
                }
                for r in readings
            ]
