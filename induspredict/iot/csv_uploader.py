"""
IndusPredict AI - CSV & Excel Telemetry Ingestion Engine.
Parses, cleans, validates, and ingests historical industrial datasets (e.g. MetroPT-3).
All uploaded readings are strictly persisted with source="uploaded".
Supports time-series playback into live charts.
"""

from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import Machine, Sensor, SensorReading
from induspredict.db.session import get_db_session

logger = get_logger("iot.csv_uploader")


class CSVTelemetryUploader(QObject):
    """
    Ingests and parses time-series CSV and Excel files into the industrial database.
    Provides validation, missing-value imputation, and replay playback.
    """

    upload_progress = Signal(int, int)  # current, total
    upload_completed = Signal(dict)     # summary metrics
    playback_frame = Signal(dict)       # frame during playback
    error_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback_running = False
        self._playback_thread: Optional[threading.Thread] = None

    def parse_and_validate_file(self, file_path: str) -> Tuple[bool, pd.DataFrame, str]:
        """
        Inspect CSV or Excel file structure, check columns, and handle missing values.
        Returns: (success, dataframe, message)
        """
        path = Path(file_path)
        if not path.exists():
            return False, pd.DataFrame(), f"File does not exist: {file_path}"

        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(file_path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                return False, pd.DataFrame(), f"Unsupported file extension: {path.suffix}. Use .csv or .xlsx"
        except Exception as e:
            return False, pd.DataFrame(), f"Failed to read file: {e}"

        if df.empty:
            return False, pd.DataFrame(), "Uploaded file contains no rows."

        # Normalize column names: strip whitespace, lowercase
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Detect timestamp column
        ts_candidates = ["timestamp", "time", "date", "datetime", "record_time"]
        ts_col = next((c for c in ts_candidates if c in df.columns), None)

        if ts_col:
            try:
                df["parsed_timestamp"] = pd.to_datetime(df[ts_col])
            except Exception as e:
                df["parsed_timestamp"] = [utc_now()] * len(df)
        else:
            df["parsed_timestamp"] = [utc_now()] * len(df)

        # Detect sensor columns
        known_sensors = ["temperature", "vibration", "pressure", "motor_speed", "current", "torque"]
        present_sensors = [s for s in known_sensors if any(s in col for col in df.columns)]

        if not present_sensors:
            # Check if tabular format with 'sensor_type' and 'value' columns
            if "sensor_type" in df.columns and "value" in df.columns:
                present_sensors = ["generic_tabular"]
            else:
                return False, df, f"No recognizable sensor columns found. Expected columns such as: {known_sensors}"

        # Clean missing values using forward-fill then backward-fill
        initial_missing = df.isnull().sum().sum()
        df = df.ffill().bfill()

        msg = f"Validated {len(df)} rows across sensors {present_sensors}. (Imputed {initial_missing} missing values)"
        logger.info(msg)
        return True, df, msg

    def import_dataframe_to_db(self, df: pd.DataFrame, machine_code: str) -> Dict[str, Any]:
        """
        Batch insert parsed dataframe records into SQLite database.
        Ensures all readings are explicitly flagged with source='uploaded'.
        """
        with get_db_session() as session:
            machine = session.query(Machine).filter(Machine.machine_code == machine_code).first()
            if not machine:
                return {"success": False, "error": f"Machine '{machine_code}' not found in database."}

            # Map sensors
            sensors_map = {s.sensor_type: s for s in machine.sensors}
            imported_count = 0
            batch_size = 500
            readings_batch = []

            # Check if tabular or wide format
            if "sensor_type" in df.columns and "value" in df.columns:
                for idx, row in df.iterrows():
                    stype = str(row["sensor_type"]).lower()
                    sensor = sensors_map.get(stype)
                    if not sensor:
                        continue

                    val = float(row["value"])
                    ts = row.get("parsed_timestamp", utc_now())
                    if isinstance(ts, pd.Timestamp):
                        ts = ts.to_pydatetime()

                    is_anom = val >= sensor.warning_threshold
                    reading = SensorReading(
                        machine_id=machine.id,
                        sensor_id=sensor.id,
                        timestamp=ts,
                        value=round(val, 2),
                        source="uploaded",  # Explicitly identified
                        is_anomaly=is_anom,
                        quality_flag="GOOD",
                    )
                    readings_batch.append(reading)
                    imported_count += 1

                    if len(readings_batch) >= batch_size:
                        session.bulk_save_objects(readings_batch)
                        session.flush()
                        readings_batch = []
                        self.upload_progress.emit(imported_count, len(df))
            else:
                # Wide format: iterate rows and sensor columns
                known_sensors = ["temperature", "vibration", "pressure", "motor_speed", "current", "torque"]
                cols_to_sensor = {}
                for col in df.columns:
                    for s in known_sensors:
                        if s in col and s in sensors_map:
                            cols_to_sensor[col] = sensors_map[s]
                            break

                total_cells = len(df) * len(cols_to_sensor)
                current_cell = 0

                for idx, row in df.iterrows():
                    ts = row.get("parsed_timestamp", utc_now())
                    if isinstance(ts, pd.Timestamp):
                        ts = ts.to_pydatetime()

                    for col, sensor in cols_to_sensor.items():
                        try:
                            val = float(row[col])
                            if np.isnan(val):
                                continue
                        except (ValueError, TypeError):
                            continue

                        is_anom = val >= sensor.warning_threshold
                        reading = SensorReading(
                            machine_id=machine.id,
                            sensor_id=sensor.id,
                            timestamp=ts,
                            value=round(val, 2),
                            source="uploaded",  # Explicitly identified
                            is_anomaly=is_anom,
                            quality_flag="GOOD",
                        )
                        readings_batch.append(reading)
                        imported_count += 1
                        current_cell += 1

                    if len(readings_batch) >= batch_size:
                        session.bulk_save_objects(readings_batch)
                        session.flush()
                        readings_batch = []
                        self.upload_progress.emit(current_cell, total_cells)

            if readings_batch:
                session.bulk_save_objects(readings_batch)
                session.commit()

            summary = {
                "success": True,
                "machine_code": machine_code,
                "readings_imported": imported_count,
                "total_rows": len(df),
                "source": "uploaded",
            }
            self.upload_completed.emit(summary)
            logger.info(f"Successfully imported {imported_count} telemetry points for machine '{machine_code}'.")
            return summary

    def start_playback(self, df: pd.DataFrame, machine_code: str, speed_multiplier: float = 1.0) -> None:
        """Stream dataframe rows into live chart playback at given playback speed."""
        self.stop_playback()
        self._playback_running = True
        self._playback_thread = threading.Thread(
            target=self._run_playback_loop,
            args=(df, machine_code, speed_multiplier),
            daemon=True,
        )
        self._playback_thread.start()

    def stop_playback(self) -> None:
        """Stop active playback."""
        self._playback_running = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

    def _run_playback_loop(self, df: pd.DataFrame, machine_code: str, speed: float) -> None:
        delay = max(0.05, 1.0 / max(0.1, speed))
        known_sensors = ["temperature", "vibration", "pressure", "motor_speed", "current", "torque"]

        for idx, row in df.iterrows():
            if not self._playback_running:
                break

            sensors_data = {}
            for s in known_sensors:
                col = next((c for c in df.columns if s in c), None)
                if col and not pd.isna(row[col]):
                    val = float(row[col])
                    sensors_data[s] = {"value": round(val, 2)}

            frame = {
                "machine_code": machine_code,
                "timestamp": str(row.get("parsed_timestamp", utc_now())),
                "source": "uploaded",  # Clearly identified
                "sensors": sensors_data,
            }
            self.playback_frame.emit(frame)
            time.sleep(delay)
